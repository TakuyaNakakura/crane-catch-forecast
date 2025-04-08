#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 必要なライブラリのインポート
import pandas as pd  # データフレーム操作用
import numpy as np  # 数値計算用
import matplotlib

# Matplotlibのバックエンドを設定（GUIがない環境でも動作するよう）
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt  # データ可視化用

# scikit-learnから機械学習関連のモジュールをインポート
from sklearn.ensemble import RandomForestRegressor  # ランダムフォレスト回帰モデル
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    explained_variance_score,
)  # 評価指標
from sklearn.model_selection import TimeSeriesSplit  # 時系列データ向けの交差検証
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
)  # 特徴量の標準化とカテゴリ変数のエンコード
from sklearn.compose import (
    ColumnTransformer,
)  # 複数の特徴変換を組み合わせるためのツール
from sklearn.pipeline import Pipeline  # 前処理とモデルを組み合わせるためのツール

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression


#############################
# 前処理と特徴量エンジニアリング
#############################
def preprocess_df(df, drop_missing=True, required_cols=None):
    """
    DataFrame の前処理:
      - Unnamed列の削除
      - 「OP価格」から「¥」および「,」を除去してfloat型に変換
      - 「入荷週」から「週」を除去してfloat型に変換
      - 「ジャンル」の欠損値を「未分類」で埋める
      - 「タイトル」の欠損値を「不明」で埋める
      - 必須カラムの欠損値がある行は削除（required_cols 指定時）

    Parameters:
        df (pandas.DataFrame): 前処理するデータフレーム
        drop_missing (bool): 欠損値を含む行を削除するかどうか
        required_cols (list): 欠損値をチェックする列名のリスト

    Returns:
        pandas.DataFrame: 前処理済みのデータフレーム
    """
    # 元のデータフレームを変更しないためにコピーを作成
    df = df.copy()

    # 'Unnamed'で始まる列（CSVインポート時に自動生成される可能性がある列）を削除
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    # OP価格列の前処理：¥記号と,（カンマ）を削除して数値化
    if "OP価格" in df.columns:
        # 価格列の前処理をより堅牢に
        df["OP価格"] = (
            df["OP価格"]
            .astype(str)
            .str.replace("¥", "", regex=False)  # ¥記号を削除
            .str.replace(",", "", regex=False)  # カンマを削除
            .str.replace("\\\\", "", regex=True)  # バックスラッシュを削除
            .str.replace(" ", "", regex=False)  # スペースを削除
            .str.replace("　", "", regex=False)  # 全角スペースを削除
        )

        # 数値に変換できない値を処理
        def safe_float_convert(x):
            try:
                return float(x)
            except (ValueError, TypeError):
                # 変換できない場合はNaNを返す
                return np.nan

        df["OP価格"] = df["OP価格"].apply(safe_float_convert)

    # 入荷週列の前処理：「週」という文字を削除して数値化
    if "入荷週" in df.columns:
        df["入荷週"] = (
            df["入荷週"].astype(str).str.replace("週", "", regex=False)  # 「週」を削除
        )

        # 数値に変換できない値を処理
        df["入荷週"] = df["入荷週"].apply(safe_float_convert)

    # ジャンル列の欠損値を「未分類」で埋める
    if "ジャンル" in df.columns:
        df["ジャンル"] = df["ジャンル"].fillna("未分類")

    # タイトル列の欠損値を「不明」で埋める
    if "タイトル" in df.columns:
        df["タイトル"] = df["タイトル"].fillna("不明")

    # 欠損値を含む行を削除するオプション
    if drop_missing:
        # required_colsが指定されていない場合のデフォルト値
        if required_cols is None:
            required_cols = ["入数", "OP価格", "入荷週", "ジャンル", "タイトル", "IOQ"]
        # 指定された列に欠損値がある行を削除
        df = df.dropna(subset=required_cols)

    return df


def add_features(df):
    """
    追加の特徴量エンジニアリング
      - 単価 = OP価格 / 入数 を追加する

    Parameters:
        df (pandas.DataFrame): 特徴量を追加するデータフレーム

    Returns:
        pandas.DataFrame: 新しい特徴量が追加されたデータフレーム
    """
    # 元のデータフレームを変更しないためにコピーを作成
    df = df.copy()

    # 「単価」特徴量の作成: OP価格を入数で割る
    # 注意: 入数が0の場合を避けるため、ごく小さい値を足している
    df["単価"] = df["OP価格"] / (df["入数"] + 1e-6)

    return df


#############################
# メイン処理
#############################
def main():
    #############################
    # Step 1: 学習データの読み込み＆前処理、特徴量エンジニアリング
    #############################
    # 8ヶ月分の学習データを読み込む
    train_csv = "data/train_8_month.csv"
    df_train = pd.read_csv(train_csv)

    # 前処理を適用
    df_train = preprocess_df(df_train, drop_missing=True)

    # 新しい特徴量（単価）を追加
    df_train = add_features(df_train)

    # データの先頭部分を表示して確認
    print("=== Training Data Head ===")
    print(df_train.head())

    #############################
    # Step 2: ターゲット変数の分布確認（ばらつきチェック）
    #############################
    # 予測対象の列を定義
    target_cols = ["IOQ"]

    # ターゲット変数の基本統計量を表示
    print("\n=== Target Distribution ===")
    print(df_train[target_cols].describe())

    #############################
    # Step 3: 特徴量の分布確認
    # ※ここでは、元の特徴量に加えて追加した「単価」を含む
    #############################
    # 使用する特徴量の列を定義
    # 数値特徴量と、カテゴリ特徴量（ジャンル、タイトル）を分けて定義
    numeric_features = ["入数", "OP価格", "入荷週"]
    categorical_features = ["ジャンル", "タイトル"]
    feature_cols = numeric_features + categorical_features

    # 数値特徴量の基本統計量を表示
    print("\n=== Numeric Feature Distribution ===")
    print(df_train[numeric_features].describe())

    # カテゴリ特徴量の分布を表示
    print("\n=== Categorical Feature Distribution ===")
    print("ジャンルの分布 (上位10カテゴリ):")
    print(df_train["ジャンル"].value_counts().head(10))
    print(f"ジャンルの総カテゴリ数: {df_train['ジャンル'].nunique()}")

    print("\nタイトルの分布 (上位10カテゴリ):")
    print(df_train["タイトル"].value_counts().head(10))
    print(f"タイトルの総カテゴリ数: {df_train['タイトル'].nunique()}")

    #############################
    # Step 4: 学習データの定義 & 特徴量の変換パイプライン作成
    #############################
    # 特徴量とターゲット変数を分離
    X_train = df_train[feature_cols]  # 特徴量
    y_train = df_train[target_cols]  # ターゲット変数

    # 数値特徴量と、カテゴリ特徴量に対する変換器を定義
    # 数値特徴量に対してはStandardScaler、カテゴリ特徴量に対してはOneHotEncoderを使用
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    # 各モデルの定義
    xgb_model = xgb.XGBRegressor(
        objective="reg:squarederror", eval_metric="rmse", use_label_encoder=False
    )
    lgb_model = lgb.LGBMRegressor()
    catboost_model = CatBoostRegressor(verbose=0)  # ログ出力を抑制

    # StackingRegressor の構築（基本学習器として各モデル、最終的なメタ学習器として線形回帰を利用）
    estimators = [("xgb", xgb_model), ("lgb", lgb_model), ("cat", catboost_model)]

    stacking_model = StackingRegressor(
        estimators=estimators, final_estimator=LinearRegression()
    )

    # モデルとパイプラインを作成
    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", stacking_model),
        ]
    )

    #############################
    # Step 5: 時系列クロスバリデーションによる評価（ハイパーパラメータ調整）
    #############################
    # 時系列データ用の交差検証分割器を作成（10分割）
    tscv = TimeSeriesSplit(n_splits=10)
    fold = 1

    # 各評価指標を格納する辞書を初期化
    cv_metrics = {
        "mse_ioq": [],
        "rmse_ioq": [],
        "mae_ioq": [],
        "r2_ioq": [],
        "evs_ioq": [],
    }

    print("\n=== Cross Validation (8 months data) ===")

    # 時系列交差検証を実行（時間的に古いデータで学習し、新しいデータで検証）
    for train_index, val_index in tscv.split(X_train):
        # 各分割でのトレーニングデータと検証データを取得
        X_cv_train, X_cv_val = X_train.iloc[train_index], X_train.iloc[val_index]
        y_cv_train, y_cv_val = y_train.iloc[train_index], y_train.iloc[val_index]

        # パイプラインでモデルを学習（前処理とモデル学習を一緒に行う）
        model_pipeline.fit(X_cv_train, y_cv_train.values.ravel())

        # 検証データで予測
        y_cv_pred = model_pipeline.predict(X_cv_val)

        # -----評価指標の計算: 「IOQ」列-----
        # MSE (Mean Squared Error): 平均二乗誤差
        mse_ioq = mean_squared_error(y_cv_val, y_cv_pred)
        # RMSE (Root Mean Squared Error): 平方根平均二乗誤差
        rmse_ioq = np.sqrt(mse_ioq)
        # MAE (Mean Absolute Error): 平均絶対誤差
        mae_ioq = mean_absolute_error(y_cv_val, y_cv_pred)
        # R² (R-squared): 決定係数（1に近いほど良い）
        r2_ioq = r2_score(y_cv_val, y_cv_pred)
        # EVS (Explained Variance Score): 説明分散スコア（1に近いほど良い）
        evs_ioq = explained_variance_score(y_cv_val, y_cv_pred)

        # 評価結果を表示
        print(f"\n--- Fold {fold} ---")
        print(
            f"IOQ: MSE: {mse_ioq:.3f}, RMSE: {rmse_ioq:.3f}, MAE: {mae_ioq:.3f}, R²: {r2_ioq:.3f}, EVS: {evs_ioq:.3f}"
        )

        # 評価指標を辞書に追加（後で平均を計算するため）
        cv_metrics["mse_ioq"].append(mse_ioq)
        cv_metrics["rmse_ioq"].append(rmse_ioq)
        cv_metrics["mae_ioq"].append(mae_ioq)
        cv_metrics["r2_ioq"].append(r2_ioq)
        cv_metrics["evs_ioq"].append(evs_ioq)
        fold += 1

    # 全ての分割での評価指標の平均を計算して表示
    print("\n=== Average Cross Validation Metrics ===")
    print(
        f"IOQ: MSE: {np.mean(cv_metrics['mse_ioq']):.3f}, RMSE: {np.mean(cv_metrics['rmse_ioq']):.3f}, "
        f"MAE: {np.mean(cv_metrics['mae_ioq']):.3f}, R²: {np.mean(cv_metrics['r2_ioq']):.3f}, EVS: {np.mean(cv_metrics['evs_ioq']):.3f}"
    )

    #############################
    # Step 6: 8ヶ月全データで最終モデルの学習
    #############################
    # クロスバリデーションで得られた最適なハイパーパラメータを使用して最終モデルを学習
    model_pipeline.fit(X_train, y_train.values.ravel())
    print("\n=== Final Model Training Completed on Entire 8-month Data ===")

    #############################
    # Step 7: テストデータ（2ヶ月分）での評価
    #############################
    # テストデータの読み込みと前処理
    test_csv = "data/test_2_month.csv"
    df_test = pd.read_csv(test_csv)
    df_test = preprocess_df(df_test, drop_missing=True)
    df_test = add_features(df_test)

    # 必要な列がテストデータに存在するか確認
    for col in feature_cols + target_cols:
        if col not in df_test.columns:
            raise ValueError(f"Column '{col}' is not found in test data.")

    # テストデータから特徴量とターゲットを分離
    X_test = df_test[feature_cols]
    y_test = df_test[target_cols]

    # パイプラインを使用して予測（前処理も含む）
    y_pred = model_pipeline.predict(X_test)

    # テストデータでの評価指標を計算して表示
    print("\n=== Test Data Evaluation Metrics ===")
    # 各評価指標を計算
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    evs = explained_variance_score(y_test, y_pred)

    # 結果を表示
    print(f"\n--- IOQ ---")
    print(f"MSE: {mse:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"MAE: {mae:.3f}")
    print(f"R²: {r2:.3f}")
    print(f"Explained Variance: {evs:.3f}")

    # 予測結果と実際の値の差（誤差）を計算し、CSVファイルに保存
    df_test["Predicted IOQ"] = y_pred  # IOQの予測値

    # 要求された追加の列を計算
    df_test["Predicted IOQ Packages"] = df_test["Predicted IOQ"] / df_test["入数"]
    df_test["Absolute IOQ Error"] = (df_test["Predicted IOQ"] - df_test["IOQ"]).abs()

    # IOQケース数（実際のIOQを入数で割った値）とPredicted IOQ Packagesの絶対誤差
    df_test["Absolute IOQ Packages Error"] = (
        df_test["IOQ"] / df_test["入数"] - df_test["Predicted IOQ Packages"]
    ).abs()

    # Adjusted Predicted IOQ Packagesの計算
    def adjust_packages(value):
        if value >= 1.0:
            # 1以上なら小数点第0位（整数部分）に切り捨て
            return np.floor(value)
        elif value >= 0.8:
            # 0.8以上1未満なら1
            return 1.0
        elif value >= 0.4:
            # 0.4以上0.8未満なら0.5
            return 0.5
        else:
            # 0.4未満なら0
            return 0.0

    df_test["Adjusted Predicted IOQ Packages"] = df_test[
        "Predicted IOQ Packages"
    ].apply(adjust_packages)

    # Adjusted Absolute IOQ Errorの計算（IOQとAdjusted Predicted IOQ Packagesの絶対誤差）
    # IOQを入数で割って実際のパッケージ数に変換し、調整後の予測パッケージ数との絶対誤差を計算
    df_test["Adjusted Absolute IOQ Packages Error"] = (
        df_test["IOQ"] / df_test["入数"] - df_test["Adjusted Predicted IOQ Packages"]
    ).abs()

    # 結果をCSVに保存（BOMありUTF-8で日本語を正しく保存）
    df_test.to_csv("result.csv", index=False, encoding="utf-8-sig")
    print("\nResults with per-sample errors have been saved to result.csv.")

    #############################
    # Step 8: テストデータの可視化
    #############################
    # サンプル数を取得（グラフのアノテーションに使用）
    total_samples = len(df_test)

    # IOQの散布図（実際の値 vs 予測値）
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, color="blue", alpha=0.6, edgecolors="k")
    # 理想的な予測線（45度線）をプロット
    plt.plot(
        [y_test.values.min(), y_test.values.max()],
        [y_test.values.min(), y_test.values.max()],
        "r--",
        lw=2,
    )
    plt.xlabel("True IOQ")
    plt.ylabel("Predicted IOQ")
    plt.title("True IOQ vs Predicted IOQ")
    plt.grid(True)
    # サンプル数をグラフに注釈として追加
    plt.annotate(
        f"n = {total_samples}",
        xy=(0.05, 0.95),
        xycoords="axes fraction",
        fontsize=12,
        color="black",
    )
    # 画像として保存
    plt.savefig("ioq_scatter.png", bbox_inches="tight")
    plt.show(block=False)
    plt.pause(2)  # 2秒間表示してから閉じる
    plt.close()

    #############################
    # Step 9: 将来予測用データで予測
    #############################
    try:
        # 将来データの読み込みを試みる
        future_csv = "data/2025_8.csv"  # 将来予測用データのファイル名を2025_8.csvに変更

        print(f"\nTrying to read future data file: {future_csv}")
        try:
            # まずUTF-8で試す
            df_future = pd.read_csv(future_csv, encoding="utf-8")
            print("Successfully read file with UTF-8 encoding")
        except UnicodeDecodeError:
            try:
                # UTF-8で失敗した場合はShift-JISを試す
                df_future = pd.read_csv(future_csv, encoding="shift-jis")
                print("Successfully read file with Shift-JIS encoding")
            except UnicodeDecodeError:
                try:
                    # Shift-JISで失敗した場合はCP932を試す
                    df_future = pd.read_csv(future_csv, encoding="cp932")
                    print("Successfully read file with CP932 encoding")
                except UnicodeDecodeError:
                    # それでも失敗した場合はISO-8859-1を試す
                    df_future = pd.read_csv(future_csv, encoding="iso-8859-1")
                    print("Successfully read file with ISO-8859-1 encoding")

        print("\nColumn names in future data:")
        print(df_future.columns.tolist())

        # 列名が文字化けしている場合は、適切な列名に変更
        # 2025_8.csvの列名マッピング（iso-8859-1エンコーディングで読み込んだ場合）
        column_mapping = {
            "Ô": "番号",
            "[J[": "メーカー",
            "^Cg": "タイトル",
            "L": "キャラ",
            "¤ i ¼": "商 品 名",
            "W": "ジャンル",
            "TuW": "サブジャンル",
            "ü": "入数",
            "OP¿i": "OP価格",
            "ü×T": "入荷週",
        }

        # 既存の列名を確認し、マッピングが必要な列があれば変更
        df_future = df_future.rename(columns=column_mapping)

        print("\nColumn names after mapping:")
        print(df_future.columns.tolist())

        # 必須列のチェックを先に行う
        missing_cols = [col for col in feature_cols if col not in df_future.columns]
        if missing_cols:
            print(f"\nWarning: Missing columns in future data: {missing_cols}")
            print("Future prediction will be skipped.")
            raise ValueError(f"Missing required columns: {missing_cols}")

        # 将来データは特徴量のみ必須（ターゲット変数はまだ存在しない）
        future_required_cols = ["入数", "OP価格", "入荷週", "ジャンル", "タイトル"]
        df_future = preprocess_df(
            df_future, drop_missing=True, required_cols=future_required_cols
        )
        df_future = add_features(df_future)

        # 将来データから特徴量を抽出
        X_future = df_future[feature_cols]

        # パイプラインを使用して予測（前処理も含む）
        future_pred = model_pipeline.predict(X_future)

        # 予測結果をデータフレームに追加
        df_future["Predicted IOQ"] = future_pred

        # 結果をCSVに保存
        df_future.to_csv("future_prediction.csv", index=False, encoding="utf-8-sig")
        print("\nFuture predictions saved to future_prediction.csv.")

        #############################
        # Step 10: 将来予測結果の可視化
        #############################
        # X軸の設定（日付列があれば使用、なければインデックスを使用）
        if "日付" in df_future.columns:
            x_axis = pd.to_datetime(df_future["日付"])
            x_label = "Date"
        else:
            x_axis = df_future.index
            x_label = "Index"

        # IOQの予測値を時系列グラフとして表示
        plt.figure(figsize=(10, 6))
        plt.plot(x_axis, df_future["Predicted IOQ"], marker="o", linestyle="-")
        plt.xlabel(x_label)
        plt.ylabel("Predicted IOQ")
        plt.title("Future Prediction: IOQ")
        plt.grid(True)
        plt.savefig("future_ioq_prediction.png", bbox_inches="tight")
        plt.show(block=False)
        plt.pause(2)
        plt.close()

    except FileNotFoundError as e:
        print(f"\nWarning: Future data file not found: {e}")
        print("Future prediction will be skipped.")
    except ValueError as e:
        print(f"\nWarning: {e}")
        print("Check that your future data file contains all required columns.")
    except Exception as e:
        print(f"\nError during future prediction: {e}")
        print("Future prediction will be skipped.")


# メインプログラムの実行（このスクリプトが直接実行された場合のみ実行）
if __name__ == "__main__":
    main()
