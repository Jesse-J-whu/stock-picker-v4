"""
鸭口选股策略 V4
========================
在 V3 六大指标 × 三周期基础上，新增核心条件：
  【日线 + 周线 + 月线 当前均处于鸭口形态（BOLL 开口扩张）】
即要求最新一根 K 线本身就在 BOLL 开口扩张状态，
而不仅仅是历史某段时间内曾经出现过。

─────────────────────────────────────────────
新增条件（V4 专属）：
  BOLL 当前鸭口（三周期同时满足）
  日线：当前 UB↑ & MID↑ & LB↓
  周线：当前 UB↑ & MID↑ & LB↓
  月线：当前 UB↑ & MID↑ & LB↓

原有六大指标（V3 全部保留，条件不变）：

一、BOLL（布林带开口扩张）
    月线：12个月内出现过 UB↑ & BOLL↑ & LB↓
    周线：26周内出现过   UB↑ & BOLL↑ & LB↓
    日线：22日内出现过   UB↑ & BOLL↑ & LB↓

二、MACD（DIF金叉且持续在DEA上方）
    月线：12个月内出现月线DIF↑穿DEA且此后DIF始终>=DEA（零轴上下均可）
    周线：26周内出现   周线零轴上 DIF↑穿DEA且此后DIF始终>=DEA
    日线：22日内出现   日线DIF↑穿DEA且此后DIF始终>=DEA

三、OBV
    月线/周线/日线：OBV > MA(OBV, 20)

四、DMA（月线+周线，不含日线）
    月线/周线：DMA_DIF > DMA_DIFMA
    （DMA_DIF = MA(close,10)-MA(close,50), DMA_DIFMA = MA(DIF,10)）

五、AMO放量（成交量条件）
    52周内任意一周 vol > 前周 3倍
    26周内任意一周 vol > 前周 1.5倍（两者同时满足）
    22日内任意一日 vol > 前日 1.5倍

六、KDJ金叉
    月线：24个月内出现 J↑穿K穿D（J上穿K且K上穿D）
    周线：26周内出现   J↑穿K穿D
    日线：22日内出现   J↑穿K穿D
"""

import numpy as np
import pandas as pd
import json
import os
import sys
import re
import requests
from datetime import datetime
from jinja2 import Template
import time

# ============================================================
# HTTP 基础设施
# ============================================================

HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/120.0.0.0 Safari/537.36'
    ),
}
SESSION = requests.Session()
SESSION.headers.update(HEADERS)


# ============================================================
# 数据获取层
# ============================================================

def get_all_a_stocks():
    """通过腾讯实时行情批量探测有效A股"""
    print("[1/4] 获取A股股票列表...")

    code_ranges = []
    code_ranges += [f"sz{str(i).zfill(6)}" for i in range(1, 1000)]
    code_ranges += [f"sz{str(i).zfill(6)}" for i in range(2001, 3000)]
    code_ranges += [f"sz{str(i).zfill(6)}" for i in range(300001, 302000)]
    code_ranges += [f"sh{str(i).zfill(6)}" for i in range(600000, 602000)]
    code_ranges += [f"sh{str(i).zfill(6)}" for i in range(603000, 604000)]
    code_ranges += [f"sh{str(i).zfill(6)}" for i in range(605000, 606000)]
    code_ranges += [f"sh{str(i).zfill(6)}" for i in range(688001, 690000)]

    all_stocks = []
    batch_size = 80

    for i in range(0, len(code_ranges), batch_size):
        batch = code_ranges[i:i + batch_size]
        query = ','.join(batch)
        url = f"https://qt.gtimg.cn/q={query}"
        try:
            resp = SESSION.get(url, timeout=20)
            if resp.status_code != 200:
                continue
            text = resp.text
            for entry in text.split(';'):
                entry = entry.strip()
                if not entry:
                    continue
                match = re.search(r'v_(\w+)="(\d+)~(.+?)~(\d+)~([^~]*)~', entry)
                if not match:
                    continue
                name = match.group(3).strip()
                code = match.group(4)
                price_str = match.group(5)
                if not name or not code or len(code) != 6:
                    continue
                if 'ST' in name or '退' in name or 'PT' in name:
                    continue
                try:
                    price = float(price_str)
                    if price <= 0:
                        continue
                except (ValueError, TypeError):
                    continue
                all_stocks.append({'代码': code, '名称': name})
        except Exception:
            continue

        if (i // batch_size) % 20 == 0 and i > 0:
            print(f"    已探测 {i}/{len(code_ranges)}，有效 {len(all_stocks)} 只...")
        time.sleep(0.05)

    df = pd.DataFrame(all_stocks)
    if df.empty:
        print("  股票列表获取失败!")
        return df
    df = df.drop_duplicates(subset='代码').reset_index(drop=True)
    print(f"  共 {len(df)} 只股票待筛选")
    return df


def _fetch_kline(symbol, period, count):
    """
    通用K线获取（腾讯财经前复权接口）
    period: 'week' / 'month' / 'day'
    返回 DataFrame(date, open, close, high, low, vol) 或空 DataFrame
    """
    period_map = {'week': 'qfqweek', 'month': 'qfqmonth', 'day': 'qfqday'}
    qfq_key = period_map.get(period, f'qfq{period}')

    url = (
        f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?"
        f"_var=kline_{period}qfq&param={symbol},{period},,,{count},qfq"
    )
    try:
        resp = SESSION.get(url, timeout=20)
        if resp.status_code != 200:
            return pd.DataFrame()
        text = resp.text.strip()
        if '=' in text:
            text = text.split('=', 1)[1]
        data = json.loads(text)
        if data.get('code') != 0:
            return pd.DataFrame()
        stock_data = data.get('data', {})
        if not stock_data:
            return pd.DataFrame()
        first_key = list(stock_data.keys())[0]
        klines = stock_data[first_key].get(qfq_key, [])
        if not klines:
            return pd.DataFrame()
        rows = []
        for k in klines:
            if len(k) >= 6:
                try:
                    rows.append({
                        'date':  k[0],
                        'open':  float(k[1]),
                        'close': float(k[2]),
                        'high':  float(k[3]),
                        'low':   float(k[4]),
                        'vol':   float(k[5]),
                    })
                except (ValueError, IndexError):
                    continue
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df = df.sort_values('date').reset_index(drop=True)
        return df
    except Exception:
        return pd.DataFrame()


def get_kline(stock_code, period, count):
    """统一接口：按股票代码 + 周期获取K线"""
    if stock_code.startswith(('60', '68')):
        symbol = f"sh{stock_code}"
    else:
        symbol = f"sz{stock_code}"
    return _fetch_kline(symbol, period, count)


def get_daily_display(stock_code):
    """获取最新实时行情（用于展示）"""
    if stock_code.startswith(('60', '68')):
        symbol = f"sh{stock_code}"
    else:
        symbol = f"sz{stock_code}"
    url = f"https://qt.gtimg.cn/q={symbol}"
    try:
        resp = SESSION.get(url, timeout=15)
        text = resp.text.strip()
        match = re.search(r'"(.+)"', text)
        if not match:
            return {}
        parts = match.group(1).split('~')
        if len(parts) < 40:
            return {}
        price = float(parts[3])
        prev_close = float(parts[4])
        change_pct = (price - prev_close) / prev_close * 100 if prev_close > 0 else 0
        return {
            'price':      price,
            'change_pct': round(change_pct, 2),
            'volume':     float(parts[36]) if parts[36] else 0,
            'high':       float(parts[33]) if parts[33] else price,
            'low':        float(parts[34]) if parts[34] else price,
            'open':       float(parts[5])  if parts[5]  else price,
        }
    except Exception:
        return {}


# ============================================================
# 技术指标计算工具
# ============================================================

def ema(series, n):
    return series.ewm(span=n, adjust=False).mean()

def ma(series, n):
    return series.rolling(window=n, min_periods=n).mean()

def std_dev(series, n):
    return series.rolling(window=n, min_periods=n).std(ddof=0)

def ref(series, n):
    return series.shift(n)

def exist(cond_series, n):
    """最近 n 个周期内是否出现过 True"""
    return cond_series.rolling(window=n, min_periods=1).max().astype(bool)

def cross_up(s1, s2):
    """s1 上穿 s2（金叉）"""
    return (s1 > s2) & (ref(s1, 1) <= ref(s2, 1))


# ============================================================
# 各指标计算函数
# ============================================================

def calc_boll(df, period=20):
    """
    返回：boll_cond（Series[bool]）
    条件：UB上升 & MID上升 & LB下降（鸭口扩张）
    """
    close = df['close']
    mid   = ma(close, period)
    upper = mid + 2 * std_dev(close, period)
    lower = mid - 2 * std_dev(close, period)
    cond  = (upper > ref(upper, 1)) & (mid > ref(mid, 1)) & (lower < ref(lower, 1))
    return cond


def calc_boll_current(df, period=20):
    """
    【V4 新增】判断当前（最新一根K线）是否处于鸭口形态
    返回 bool：当前 UB↑ & MID↑ & LB↓
    """
    cond = calc_boll(df, period)
    if cond.empty:
        return False
    return bool(cond.iloc[-1])


def calc_macd(df, with_zero_filter=False):
    """
    返回：macd_cross_hold（Series[bool]）
    金叉后 DIF 始终 >= DEA
    with_zero_filter=True 时，要求金叉发生时 DEA > 0（零轴上方）
    """
    close = df['close']
    dif   = ema(close, 12) - ema(close, 26)
    dea   = ema(dif, 9)

    jc = cross_up(dif, dea)
    if with_zero_filter:
        jc = jc & (dea > 0)

    dif_above = (dif >= dea).astype(float)

    result = pd.Series(False, index=df.index)
    jc_idx = df.index[jc]
    for idx in jc_idx:
        subsequent = dif_above.loc[idx:]
        if subsequent.min() >= 1.0:
            result.loc[idx:] = True
    return result


def calc_obv(df, ma_period=20):
    """OBV > MA(OBV, ma_period)"""
    close = df['close']
    vol   = df['vol']
    direction = np.sign(close.diff().fillna(0))
    obv   = (direction * vol).cumsum()
    maobv = ma(obv, ma_period)
    return obv > maobv


def calc_dma(df):
    """
    DMA 指标：DIF_DMA > DIFMA
    DIF_DMA = MA(close,10) - MA(close,50)
    DIFMA   = MA(DIF_DMA, 10)
    """
    close   = df['close']
    dif_dma = ma(close, 10) - ma(close, 50)
    difma   = ma(dif_dma, 10)
    return dif_dma > difma


def calc_amo(df_week, df_daily):
    """
    放量条件（三个同时满足）：
    A: 52周内任意一周 vol > 前周 3倍
    B: 26周内任意一周 vol > 前周 1.5倍
    C: 22日内任意一日 vol > 前日 1.5倍
    """
    wvol  = df_week['vol']
    ratio_w = wvol / ref(wvol, 1)
    cond_a = exist(ratio_w > 3,   52)
    cond_b = exist(ratio_w > 1.5, 26)

    dvol  = df_daily['vol']
    ratio_d = dvol / ref(dvol, 1)
    cond_c = exist(ratio_d > 1.5, 22)

    a = bool(cond_a.iloc[-1]) if not cond_a.empty else False
    b = bool(cond_b.iloc[-1]) if not cond_b.empty else False
    c = bool(cond_c.iloc[-1]) if not cond_c.empty else False
    return a and b and c


def calc_kdj(df, n=9, m1=3, m2=3):
    """
    KDJ 计算（经典随机指标）
    金叉条件：J 上穿 K 且 K 上穿 D（同一根K线发生）
    """
    high  = df['high']
    low   = df['low']
    close = df['close']

    low_n  = low.rolling(window=n, min_periods=1).min()
    high_n = high.rolling(window=n, min_periods=1).max()

    rsv = (close - low_n) / (high_n - low_n + 1e-9) * 100
    rsv = rsv.clip(0, 100)

    k = pd.Series(50.0, index=df.index)
    d = pd.Series(50.0, index=df.index)
    for i in range(1, len(df)):
        k.iloc[i] = k.iloc[i-1] * (1 - 1/m1) + rsv.iloc[i] * (1/m1)
        d.iloc[i] = d.iloc[i-1] * (1 - 1/m2) + k.iloc[i] * (1/m2)

    j = 3 * k - 2 * d

    jk_cross = cross_up(j, k)
    kd_cross = cross_up(k, d)
    kdj_cross = jk_cross & kd_cross
    return kdj_cross


# ============================================================
# 主策略：六大指标 + V4 新增当前三周期鸭口
# ============================================================

def apply_strategy(df_month, df_week, df_day):
    """
    传入月线、周线、日线 DataFrame（均含 date/open/close/high/low/vol）
    返回 True/False —— 当前是否满足所有条件

    V4 在 V3 基础上新增：
      当前日线、周线、月线同时处于鸭口形态（BOLL 开口扩张）
    """

    # ── 【V4 新增】当前三周期鸭口（强过滤条件）──────────────────
    # 要求当前最新一根K线，日/周/月线均处于 BOLL 开口扩张（鸭口）状态
    boll_cur_d = calc_boll_current(df_day)
    boll_cur_w = calc_boll_current(df_week)
    boll_cur_m = calc_boll_current(df_month)
    boll_current_ok = boll_cur_d and boll_cur_w and boll_cur_m

    # 快速剪枝：当前三周期鸭口不满足直接返回 False
    if not boll_current_ok:
        return False

    # ── 一、BOLL（历史窗口内出现过）──────────────────────────────
    boll_m = exist(calc_boll(df_month), 12).iloc[-1]
    boll_w = exist(calc_boll(df_week),  26).iloc[-1]
    boll_d = exist(calc_boll(df_day),   22).iloc[-1]
    boll_ok = bool(boll_m) and bool(boll_w) and bool(boll_d)

    # ── 二、MACD ─────────────────────────────────────────────────
    macd_m_series = calc_macd(df_month, with_zero_filter=False)
    macd_m = bool(exist(macd_m_series, 12).iloc[-1])

    macd_w_series = calc_macd(df_week, with_zero_filter=True)
    macd_w = bool(exist(macd_w_series, 26).iloc[-1])

    macd_d_series = calc_macd(df_day, with_zero_filter=False)
    macd_d = bool(exist(macd_d_series, 22).iloc[-1])

    macd_ok = macd_m and macd_w and macd_d

    # ── 三、OBV ──────────────────────────────────────────────────
    obv_m = bool(calc_obv(df_month).iloc[-1])
    obv_w = bool(calc_obv(df_week).iloc[-1])
    obv_d = bool(calc_obv(df_day).iloc[-1])
    obv_ok = obv_m and obv_w and obv_d

    # ── 四、DMA（仅月线+周线）────────────────────────────────────
    dma_m = bool(calc_dma(df_month).iloc[-1])
    dma_w = bool(calc_dma(df_week).iloc[-1])
    dma_ok = dma_m and dma_w

    # ── 五、AMO放量 ───────────────────────────────────────────────
    amo_ok = calc_amo(df_week, df_day)

    # ── 六、KDJ金叉 ──────────────────────────────────────────────
    kdj_m = bool(exist(calc_kdj(df_month), 24).iloc[-1])
    kdj_w = bool(exist(calc_kdj(df_week),  26).iloc[-1])
    kdj_d = bool(exist(calc_kdj(df_day),   22).iloc[-1])
    kdj_ok = kdj_m and kdj_w and kdj_d

    return boll_ok and macd_ok and obv_ok and dma_ok and amo_ok and kdj_ok


def apply_strategy_detail(df_month, df_week, df_day):
    """
    返回各子条件详情字典，用于调试/展示
    包含 V4 新增的"当前鸭口"条件
    """
    # V4 新增：当前鸭口
    boll_cur_d = calc_boll_current(df_day)
    boll_cur_w = calc_boll_current(df_week)
    boll_cur_m = calc_boll_current(df_month)

    boll_m = bool(exist(calc_boll(df_month), 12).iloc[-1])
    boll_w = bool(exist(calc_boll(df_week),  26).iloc[-1])
    boll_d = bool(exist(calc_boll(df_day),   22).iloc[-1])

    macd_m = bool(exist(calc_macd(df_month, False), 12).iloc[-1])
    macd_w = bool(exist(calc_macd(df_week,  True),  26).iloc[-1])
    macd_d = bool(exist(calc_macd(df_day,   False), 22).iloc[-1])

    obv_m  = bool(calc_obv(df_month).iloc[-1])
    obv_w  = bool(calc_obv(df_week).iloc[-1])
    obv_d  = bool(calc_obv(df_day).iloc[-1])

    dma_m  = bool(calc_dma(df_month).iloc[-1])
    dma_w  = bool(calc_dma(df_week).iloc[-1])

    amo    = calc_amo(df_week, df_day)

    kdj_m  = bool(exist(calc_kdj(df_month), 24).iloc[-1])
    kdj_w  = bool(exist(calc_kdj(df_week),  26).iloc[-1])
    kdj_d  = bool(exist(calc_kdj(df_day),   22).iloc[-1])

    return {
        'BOLL_NOW': f"月{'✓' if boll_cur_m else '✗'} 周{'✓' if boll_cur_w else '✗'} 日{'✓' if boll_cur_d else '✗'}",
        'BOLL':     f"月{'✓' if boll_m else '✗'} 周{'✓' if boll_w else '✗'} 日{'✓' if boll_d else '✗'}",
        'MACD':     f"月{'✓' if macd_m else '✗'} 周{'✓' if macd_w else '✗'} 日{'✓' if macd_d else '✗'}",
        'OBV':      f"月{'✓' if obv_m  else '✗'} 周{'✓' if obv_w  else '✗'} 日{'✓' if obv_d  else '✗'}",
        'DMA':      f"月{'✓' if dma_m  else '✗'} 周{'✓' if dma_w  else '✗'}",
        'AMO':      '✓' if amo else '✗',
        'KDJ':      f"月{'✓' if kdj_m  else '✗'} 周{'✓' if kdj_w  else '✗'} 日{'✓' if kdj_d  else '✗'}",
    }


# ============================================================
# 主流程
# ============================================================

MIN_MONTH = 30
MIN_WEEK  = 60
MIN_DAY   = 60

FETCH_MONTH = 60
FETCH_WEEK  = 130
FETCH_DAY   = 100


def run_strategy():
    print("=" * 60)
    print(f"  鸭口选股 V4 - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    stocks = get_all_a_stocks()
    if stocks.empty:
        print("无法获取股票列表，退出")
        return []

    selected = []
    total    = len(stocks)
    failed   = 0

    print(f"\n[2/4] 逐只计算策略信号（共 {total} 只）...")
    for idx, row in stocks.iterrows():
        code = row['代码']
        name = row['名称']

        if idx % 200 == 0:
            print(f"  进度: {idx}/{total} ({idx/total*100:.1f}%)")

        df_month = get_kline(code, 'month', FETCH_MONTH)
        df_week  = get_kline(code, 'week',  FETCH_WEEK)
        df_day   = get_kline(code, 'day',   FETCH_DAY)

        if (df_month.empty or len(df_month) < MIN_MONTH or
                df_week.empty  or len(df_week)  < MIN_WEEK  or
                df_day.empty   or len(df_day)   < MIN_DAY):
            failed += 1
            time.sleep(0.05)
            continue

        try:
            hit = apply_strategy(df_month, df_week, df_day)
            if hit:
                detail = apply_strategy_detail(df_month, df_week, df_day)
                selected.append({
                    'code':   code,
                    'name':   name,
                    'detail': detail,
                })
                print(f"  ★ 选中: {code} {name}")
        except Exception as e:
            failed += 1
            time.sleep(0.05)
            continue

        time.sleep(0.15)

    print(f"\n  策略计算完成: 成功 {total - failed}, 失败 {failed}")

    print(f"\n[3/4] 获取选中股票的最新行情...")
    for item in selected:
        daily = get_daily_display(item['code'])
        item.update(daily)
        time.sleep(0.1)

    print(f"\n  共选出 {len(selected)} 只股票")
    return selected


# ============================================================
# HTML 生成
# ============================================================

def generate_html(selected_stocks, output_path):
    print(f"\n[4/4] 生成展示页面...")

    template_str = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<title>鸭口选股 V4</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC',
                 'Hiragino Sans GB', 'Microsoft YaHei', sans-serif;
    background: #080c1a;
    color: #dde3ff;
    min-height: 100vh;
    padding-bottom: env(safe-area-inset-bottom);
}
.header {
    background: linear-gradient(135deg, #131836 0%, #0a0f28 100%);
    padding: 18px 16px 14px;
    border-bottom: 1px solid rgba(90, 120, 255, 0.18);
    position: sticky;
    top: 0;
    z-index: 100;
    backdrop-filter: blur(20px);
}
.header h1 {
    font-size: 21px;
    font-weight: 800;
    background: linear-gradient(90deg, #7ba4ff, #c084fc, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: 1.5px;
}
.header .meta {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: 7px;
    font-size: 12px;
    color: #6870a0;
}
.header .count {
    background: rgba(90,120,255,0.15);
    color: #8fa4ff;
    padding: 2px 10px;
    border-radius: 12px;
    font-weight: 700;
}
.tags {
    display: flex;
    flex-wrap: wrap;
    gap: 5px;
    margin-top: 9px;
}
.tag {
    font-size: 10px;
    padding: 3px 8px;
    border-radius: 5px;
    font-weight: 600;
    letter-spacing: 0.3px;
}
.tag-v4    { background: rgba(255,100,100,0.18); color: #ff7070; border: 1px solid rgba(255,100,100,0.35); }
.tag-boll  { background: rgba(100,150,255,0.12); color: #7ba4ff; }
.tag-macd  { background: rgba(52,211,153,0.10);  color: #34d399; }
.tag-obv   { background: rgba(251,191,36,0.10);  color: #fbbf24; }
.tag-dma   { background: rgba(244,114,182,0.10); color: #f472b6; }
.tag-amo   { background: rgba(34,211,238,0.10);  color: #22d3ee; }
.tag-kdj   { background: rgba(167,139,250,0.12); color: #a78bfa; }
.strategy-desc {
    background: rgba(90,120,255,0.05);
    border: 1px solid rgba(90,120,255,0.12);
    border-radius: 10px;
    padding: 11px 13px;
    margin: 10px 12px 4px;
    font-size: 11px;
    color: #6870a0;
    line-height: 1.85;
}
.strategy-desc strong { color: #a0b0ff; }
.strategy-desc .v4-highlight { color: #ff8080; font-weight: 700; }
.disclaimer {
    background: rgba(234,179,8,0.06);
    border: 1px solid rgba(234,179,8,0.14);
    border-radius: 10px;
    padding: 10px 13px;
    margin: 4px 12px 4px;
    font-size: 11px;
    color: #a89040;
    line-height: 1.5;
}
.stock-list { padding: 10px 12px; }
.stock-card {
    background: linear-gradient(135deg, rgba(20,26,60,0.85) 0%, rgba(10,14,36,0.92) 100%);
    border: 1px solid rgba(90,120,255,0.10);
    border-radius: 14px;
    padding: 14px;
    margin-bottom: 10px;
    position: relative;
    overflow: hidden;
    transition: transform 0.15s;
}
.stock-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, rgba(255,120,120,0.5), transparent);
}
.stock-card:active { transform: scale(0.985); }
.card-top {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
}
.stock-name { font-size: 17px; font-weight: 700; color: #e6eaff; }
.stock-code {
    font-size: 12px;
    color: #525880;
    margin-top: 2px;
    font-family: 'SF Mono','Fira Code',monospace;
}
.stock-price { text-align: right; }
.price-value {
    font-size: 22px;
    font-weight: 700;
    font-family: 'SF Mono','DIN Alternate',monospace;
}
.price-change { font-size: 13px; font-weight: 600; margin-top: 1px; }
.up   { color: #f43f5e; }
.down { color: #10b981; }
.flat { color: #6870a0; }
.card-bottom {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 8px;
    margin-top: 12px;
    padding-top: 11px;
    border-top: 1px solid rgba(90,120,255,0.07);
}
.metric { text-align: center; }
.metric-label { font-size: 10px; color: #525880; letter-spacing: 0.4px; }
.metric-value {
    font-size: 13px;
    color: #a0aacc;
    margin-top: 2px;
    font-family: 'SF Mono',monospace;
}
.signal-row {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
    margin-top: 10px;
    padding-top: 9px;
    border-top: 1px solid rgba(90,120,255,0.07);
}
.sig {
    font-size: 10px;
    padding: 2px 7px;
    border-radius: 4px;
    font-family: 'SF Mono',monospace;
    white-space: nowrap;
}
.sig-v4   { background: rgba(255,100,100,0.15); color: #ff8080; border: 1px solid rgba(255,100,100,0.25); }
.sig-boll { background: rgba(100,150,255,0.1); color: #7ba4ff; }
.sig-macd { background: rgba(52,211,153,0.1);  color: #34d399; }
.sig-obv  { background: rgba(251,191,36,0.1);  color: #fbbf24; }
.sig-dma  { background: rgba(244,114,182,0.1); color: #f472b6; }
.sig-amo  { background: rgba(34,211,238,0.1);  color: #22d3ee; }
.sig-kdj  { background: rgba(167,139,250,0.1); color: #a78bfa; }
.empty-state {
    text-align: center;
    padding: 60px 20px;
    color: #525880;
}
.empty-state .icon { font-size: 48px; margin-bottom: 16px; }
.empty-state p { font-size: 14px; line-height: 1.7; }
.footer {
    text-align: center;
    padding: 18px;
    font-size: 11px;
    color: #363b5a;
    border-top: 1px solid rgba(90,120,255,0.06);
    margin-top: 8px;
}
</style>
</head>
<body>
<div class="header">
    <h1>鸭口选股 V4</h1>
    <div class="meta">
        <span>{{ update_time }}</span>
        <span class="count">{{ stock_count }} 只</span>
    </div>
    <div class="tags">
        <span class="tag tag-v4">★ 当前三周期鸭口</span>
        <span class="tag tag-boll">BOLL 月/周/日</span>
        <span class="tag tag-macd">MACD 月/周/日</span>
        <span class="tag tag-obv">OBV 月/周/日</span>
        <span class="tag tag-dma">DMA 月/周</span>
        <span class="tag tag-amo">AMO 周52&amp;26 日22</span>
        <span class="tag tag-kdj">KDJ 月/周/日</span>
    </div>
</div>

<div class="strategy-desc">
    <strong>策略逻辑（V4）：</strong>
    <span class="v4-highlight">【新增】当前日线/周线/月线同时处于 BOLL 鸭口扩张状态</span>
    + 六大指标全部通过。
    BOLL月/周/日开口扩张；MACD月/周/日金叉持续；OBV月/周/日均在均线上方；
    DMA月/周 DIF&gt;DIFMA；AMO周52周≥3倍&amp;26周≥1.5倍且日22日≥1.5倍；
    KDJ月/周/日 J穿K穿D三线金叉。
</div>

<div class="disclaimer">
    本页面仅为量化策略筛选结果展示，不构成任何投资建议。股市有风险，投资需谨慎。
</div>

<div class="stock-list">
{% if stocks %}
{% for s in stocks %}
<div class="stock-card">
    <div class="card-top">
        <div>
            <div class="stock-name">{{ s.name }}</div>
            <div class="stock-code">{{ s.code }}</div>
        </div>
        <div class="stock-price">
            {% if s.price %}
            <div class="price-value {% if s.change_pct > 0 %}up{% elif s.change_pct < 0 %}down{% else %}flat{% endif %}">
                {{ "%.2f"|format(s.price) }}
            </div>
            <div class="price-change {% if s.change_pct > 0 %}up{% elif s.change_pct < 0 %}down{% else %}flat{% endif %}">
                {% if s.change_pct > 0 %}+{% endif %}{{ "%.2f"|format(s.change_pct) }}%
            </div>
            {% else %}
            <div class="price-value flat">--</div>
            {% endif %}
        </div>
    </div>
    {% if s.price %}
    <div class="card-bottom">
        <div class="metric">
            <div class="metric-label">开盘</div>
            <div class="metric-value">{{ "%.2f"|format(s.open) }}</div>
        </div>
        <div class="metric">
            <div class="metric-label">最高</div>
            <div class="metric-value">{{ "%.2f"|format(s.high) }}</div>
        </div>
        <div class="metric">
            <div class="metric-label">最低</div>
            <div class="metric-value">{{ "%.2f"|format(s.low) }}</div>
        </div>
    </div>
    {% endif %}
    {% if s.detail %}
    <div class="signal-row">
        <span class="sig sig-v4">鸭口NOW {{ s.detail.BOLL_NOW }}</span>
        <span class="sig sig-boll">BOLL {{ s.detail.BOLL }}</span>
        <span class="sig sig-macd">MACD {{ s.detail.MACD }}</span>
        <span class="sig sig-obv">OBV {{ s.detail.OBV }}</span>
        <span class="sig sig-dma">DMA {{ s.detail.DMA }}</span>
        <span class="sig sig-amo">AMO {{ s.detail.AMO }}</span>
        <span class="sig sig-kdj">KDJ {{ s.detail.KDJ }}</span>
    </div>
    {% endif %}
</div>
{% endfor %}
{% else %}
<div class="empty-state">
    <div class="icon">📊</div>
    <p>今日暂无符合策略的股票<br>策略每个交易日收盘后自动更新</p>
</div>
{% endif %}
</div>

<div class="footer">
    <p>鸭口选股 V4 · 当前三周期鸭口 + 六大指标 × 三周期 · 数据来源：腾讯财经</p>
    <p style="margin-top:4px;">每个交易日收盘后自动更新</p>
</div>
</body>
</html>"""

    template = Template(template_str)
    html = template.render(
        stocks=selected_stocks,
        stock_count=len(selected_stocks),
        update_time=datetime.now().strftime('%Y年%m月%d日 %H:%M 更新'),
    )
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"  页面已生成: {output_path}")


def save_data_json(selected_stocks, output_path):
    """保存选股结果为 JSON"""
    data = {
        'update_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'strategy': '鸭口选股 V4',
        'conditions': {
            'BOLL_NOW': '当前日线/周线/月线同时处于 BOLL 开口扩张（鸭口）状态',
            'BOLL':     'UB↑&MID↑&LB↓ 月12期/周26期/日22期内出现',
            'MACD':     '月12/周26(零上)/日22 金叉后DIF持续>=DEA',
            'OBV':      'OBV>MA(OBV,20) 月/周/日',
            'DMA':      'DIF_DMA>DIFMA 月/周',
            'AMO':      '周52≥3x & 周26≥1.5x & 日22≥1.5x',
            'KDJ':      'J穿K穿D三线金叉 月24/周26/日22',
        },
        'count': len(selected_stocks),
        'stocks': selected_stocks,
    }
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  数据已保存: {output_path}")


if __name__ == '__main__':
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'docs')
    os.makedirs(output_dir, exist_ok=True)

    results = run_strategy()

    html_path = os.path.join(output_dir, 'index.html')
    generate_html(results, html_path)

    json_path = os.path.join(output_dir, 'data.json')
    save_data_json(results, json_path)

    print(f"\n{'=' * 60}")
    print(f"  完成! 共选出 {len(results)} 只股票")
    print(f"  HTML: {html_path}")
    print(f"  JSON: {json_path}")
    print(f"{'=' * 60}")
