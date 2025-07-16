from kaggle.api.kaggle_api_extended import KaggleApi

if __name__ == "__main__":
    api = KaggleApi()
    api.authenticate()
    # 第一个参数是 data/raw 的相对路径，unzip=True 会自动解压
    api.dataset_download_files(
        'johntitor/poetry-foundation-poems',
        path='../data/raw',
        unzip=True
    )
    print("✅ 下载并解压完成：data/raw/PoetryFoundationData.csv")