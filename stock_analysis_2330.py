"""
台積電(2330.TW)股票資料分析 —— 新手教學腳本
=================================================

這支腳本示範一套完整、可重複使用的股票資料分析流程：

    1. 下載歷史股價資料 (yfinance)
    2. 資料清理與檢視
    3. 敘述統計
    4. 技術指標：移動平均線、日報酬率、年化波動度、RSI
    5. 視覺化：價格走勢 + 均線、成交量、報酬率分佈
    6. 一個最簡單的均線交叉策略回測(教學用，非投資建議)

在自己的電腦上執行前，先安裝套件：
    pip install yfinance pandas numpy matplotlib

執行方式：
    python stock_analysis_2330.py

注意：本腳本僅供學習資料分析流程使用，不構成投資建議。
歷史績效不代表未來表現，過去的均線交叉策略在真實市場中
績效可能與回測結果差異很大（滑價、手續費、稅金都未計入）。
"""

import glob
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.font_manager as fm


def setup_cjk_font():
    """讓圖表裡的中文字正常顯示。

    只把字體『名稱』塞進 rcParams 常常沒用——如果 matplotlib 的字體
    快取沒收錄那個名字，會直接默默改用預設字體(中文變方框)，
    而且不一定會跳警告。這裡改成直接掃系統常見路徑，找到字型檔
    就用 font_manager 明確註冊，再用它真正的內部名稱設定 rcParams，
    這樣比較不會失敗。
    """
    candidate_paths = [
        # Windows
        r"C:\Windows\Fonts\msjh.ttc",       # 微軟正黑體
        r"C:\Windows\Fonts\msjhbd.ttc",
        r"C:\Windows\Fonts\mingliu.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        # macOS
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        # Linux
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            fm.fontManager.addfont(path)
            font_name = fm.FontProperties(fname=path).get_name()
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            print(f"[字體] 使用: {font_name} ({path})")
            return True

    # 上面列的路徑都找不到，再退一步：在系統已安裝字體裡搜尋名稱含中文/CJK關鍵字的
    for f in fm.fontManager.ttflist:
        name_lower = f.name.lower()
        if any(k in name_lower for k in ["jhenghei", "yahei", "pingfang", "heiti", "noto sans cjk", "wenquanyi", "simhei", "mingliu"]):
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = [f.name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            print(f"[字體] 使用: {f.name}")
            return True

    print("[字體] 找不到中文字體，圖表中的中文標籤可能無法正常顯示。"
          "可安裝任一中文字體(Windows通常已內建微軟正黑體，理論上不會走到這裡)。")
    return False

# ---------------------------------------------------------------------------
# 步驟 1：下載歷史股價資料
# ---------------------------------------------------------------------------
TICKER = "2330.TW"   # 台積電；美股例如 "AAPL"，其他台股例如 "2317.TW"(鴻海)
PERIOD = "2y"         # 抓取區間：1y, 2y, 5y, max ...
INTERVAL = "1d"       # 頻率：1d(日)、1wk(週)、1mo(月)


def fetch_stock_data(ticker=TICKER, period=PERIOD, interval=INTERVAL):
    """用 yfinance 下載歷史股價，回傳一個 DataFrame(欄位: Open/High/Low/Close/Volume)。

    如果環境無法連網（例如這個示範所在的沙盒環境會擋掉財經網站），
    就退而求其次，用同樣統計特性生成一份「模擬資料」，
    讓後面的分析流程仍然可以完整跑過一遍給你看結果長什麼樣子。
    在你自己的電腦上執行，這一步會直接抓到真實資料。
    """
    try:
        import yfinance as yf
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if df.empty:
            raise RuntimeError("下載結果是空的")
        # yfinance 新版本欄位可能是 MultiIndex，這裡統一攤平成單層欄位
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        print(f"[OK] 已從 yfinance 下載 {ticker} 共 {len(df)} 筆真實資料")
        return df
    except Exception as e:
        print(f"[提示] 無法連線抓取真實資料({e})，改用模擬資料示範分析流程。")
        return _generate_demo_data(period)


def _generate_demo_data(period):
    """僅在無法連網時使用：生成一份『看起來像真實股價』的模擬資料。
    使用幾何布朗運動(GBM)模擬價格路徑，並非任何真實股票的實際數字。
    """
    n_periods = {"6mo": 126, "1y": 252, "2y": 504, "5y": 1260}.get(period, 504)
    rng = np.random.default_rng(seed=42)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_periods)
    days = len(dates)

    s0 = 950.0          # 假設起始價(僅示範用)
    mu, sigma = 0.0004, 0.018   # 假設日報酬均值與波動度
    shocks = rng.normal(mu, sigma, size=days)
    close = s0 * np.exp(np.cumsum(shocks))

    daily_range = np.abs(rng.normal(0, sigma, size=days)) * close
    high = close + daily_range * rng.uniform(0.3, 1.0, size=days)
    low = close - daily_range * rng.uniform(0.3, 1.0, size=days)
    open_ = low + (high - low) * rng.uniform(0.2, 0.8, size=days)
    volume = rng.integers(15_000_000, 45_000_000, size=days)

    df = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )
    df.index.name = "Date"
    return df


# ---------------------------------------------------------------------------
# 步驟 2：資料清理與檢視
# ---------------------------------------------------------------------------
def clean_and_inspect(df):
    print("\n== 資料基本資訊 ==")
    print(df.info())

    n_missing = df.isna().sum().sum()
    if n_missing:
        print(f"[清理] 發現 {n_missing} 個缺值，使用前一筆資料補值(ffill)")
        df = df.ffill().dropna()
    else:
        print("[清理] 沒有缺值")

    # 股價資料常見異常：成交量為 0、價格為負，這裡做基本防呆
    bad_rows = df[(df["Close"] <= 0) | (df["Volume"] < 0)]
    if len(bad_rows):
        print(f"[清理] 移除 {len(bad_rows)} 筆異常資料")
        df = df.drop(bad_rows.index)

    return df


# ---------------------------------------------------------------------------
# 步驟 3：敘述統計
# ---------------------------------------------------------------------------
def describe_data(df):
    print("\n== 敘述統計(收盤價、成交量) ==")
    print(df[["Close", "Volume"]].describe().round(2))

    total_return = df["Close"].iloc[-1] / df["Close"].iloc[0] - 1
    print(f"\n區間總報酬率: {total_return:.2%}")
    print(f"期間最高價: {df['High'].max():.2f}　最低價: {df['Low'].min():.2f}")


# ---------------------------------------------------------------------------
# 步驟 4：技術指標
# ---------------------------------------------------------------------------
def add_indicators(df):
    df = df.copy()

    # 日報酬率
    df["Return"] = df["Close"].pct_change()

    # 移動平均線(短線 20 日、長線 60 日 —— 約一個月 / 一季的交易日)
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA60"] = df["Close"].rolling(60).mean()

    # 年化波動度(20 日滾動標準差 * sqrt(252))
    df["Volatility_20d"] = df["Return"].rolling(20).std() * np.sqrt(252)

    # RSI(14 日) —— 相對強弱指標，判斷超買超賣
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["RSI14"] = 100 - (100 / (1 + rs))

    return df


# ---------------------------------------------------------------------------
# 步驟 5：視覺化
# ---------------------------------------------------------------------------
def plot_analysis(df, ticker, out_path):
    setup_cjk_font()
    plt.rcParams.update({
        "font.size": 11,
        "axes.edgecolor": "#c3c2b7",
        "axes.labelcolor": "#52514e",
        "text.color": "#0b0b0b",
        "xtick.color": "#898781",
        "ytick.color": "#898781",
        "axes.grid": True,
        "grid.color": "#e1e0d9",
        "grid.linewidth": 0.8,
        "figure.facecolor": "#fcfcfb",
        "axes.facecolor": "#fcfcfb",
    })

    fig, axes = plt.subplots(
        3, 1, figsize=(11, 10), sharex=True,
        gridspec_kw={"height_ratios": [3, 1, 1]},
    )
    ax_price, ax_vol, ax_rsi = axes

    # --- 上圖:收盤價 + 均線 ---
    ax_price.plot(df.index, df["Close"], color="#2a78d6", linewidth=2, label="收盤價")
    ax_price.plot(df.index, df["MA20"], color="#eb6834", linewidth=1.5, label="20日均線")
    ax_price.plot(df.index, df["MA60"], color="#1baf7a", linewidth=1.5, label="60日均線")
    ax_price.set_title(f"{ticker} 股價走勢與均線", fontsize=13, color="#0b0b0b")
    ax_price.set_ylabel("價格")
    ax_price.legend(loc="upper left", frameon=False)
    for spine in ["top", "right"]:
        ax_price.spines[spine].set_visible(False)

    # --- 中圖:成交量 ---
    ax_vol.bar(df.index, df["Volume"], color="#2a78d6", alpha=0.5, width=1.0)
    ax_vol.set_ylabel("成交量")
    for spine in ["top", "right"]:
        ax_vol.spines[spine].set_visible(False)

    # --- 下圖:RSI ---
    ax_rsi.plot(df.index, df["RSI14"], color="#4a3aa7", linewidth=1.5)
    ax_rsi.axhline(70, color="#e34948", linewidth=1, linestyle="--", alpha=0.7)
    ax_rsi.axhline(30, color="#1baf7a", linewidth=1, linestyle="--", alpha=0.7)
    ax_rsi.set_ylabel("RSI(14)")
    ax_rsi.set_ylim(0, 100)
    for spine in ["top", "right"]:
        ax_rsi.spines[spine].set_visible(False)

    ax_rsi.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\n[圖表] 已儲存: {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 步驟 6(加分):最簡單的均線交叉策略回測 —— 純教學示範
# ---------------------------------------------------------------------------
def backtest_ma_cross(df):
    df = df.copy()
    # 訊號:20日均線 > 60日均線 時持有(1),否則空手(0)
    df["Signal"] = (df["MA20"] > df["MA60"]).astype(int)
    # 用前一天的訊號決定今天的部位(避免用到未來資訊)
    df["Position"] = df["Signal"].shift(1).fillna(0)
    df["StrategyReturn"] = df["Position"] * df["Return"]

    buy_hold = (1 + df["Return"]).prod() - 1
    strategy = (1 + df["StrategyReturn"]).prod() - 1

    print("\n== 均線交叉策略回測(未計手續費、稅金、滑價，僅供教學) ==")
    print(f"買進持有報酬率: {buy_hold:.2%}")
    print(f"均線交叉策略報酬率: {strategy:.2%}")
    return df


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    raw = fetch_stock_data()
    clean = clean_and_inspect(raw)
    describe_data(clean)
    enriched = add_indicators(clean)
    plot_analysis(enriched, TICKER, "stock_analysis_2330.png")
    backtest_ma_cross(enriched)

    # 把整理好的資料另存一份 CSV，方便之後用 Excel 或其他工具查看
    enriched.to_csv("stock_analysis_2330_processed.csv", encoding="utf-8-sig")
    print("[資料] 已儲存整理後的資料: stock_analysis_2330_processed.csv")
