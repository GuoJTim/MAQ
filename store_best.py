import os
import glob
import csv
import argparse

from tensorboard.backend.event_processing import event_accumulator

def export_all_scalars_to_csv_append(folder_path, csv_filename="experiment.csv"):
    """
    1) 搜尋指定資料夾下的所有 .tfevents 檔。
    2) 讀取所有 scalar (tag, step, value)。
    3) 以 'append' (a) 模式將新資料追加到同一個 CSV 檔底部。
    """

    # 尋找所有 tfevents 檔案
    event_files = glob.glob(os.path.join(folder_path, "events.out.tfevents.*"))
    if not event_files:
        print(f"找不到任何 .tfevents 檔案於: {folder_path}")
        return

    # 預計輸出 CSV 檔的路徑
    csv_path = os.path.join(folder_path, csv_filename)

    # 檢查 csv 檔案是否已存在（用來決定是否需要寫入表頭）
    file_exists = os.path.isfile(csv_path)

    # 以 append 模式開啟 CSV
    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # 若是第一次建立檔案，寫入表頭
        if not file_exists:
            writer.writerow(["event_file", "tag", "step", "value"])

        # 逐一讀取每個 tfevents 檔案
        for ef in event_files:
            print(f"正在讀取檔案: {ef}")
            ea = event_accumulator.EventAccumulator(ef)
            ea.Reload()

            scalar_tags = ea.Tags().get("scalars", [])
            for tag in scalar_tags:
                events = ea.Scalars(tag)
                for e in events:
                    writer.writerow([os.path.basename(ef), tag, e.step, e.value])

    print(f"\n已將 scalar 資料追加寫入: {csv_path}")

# -----------------------
# 範例呼叫
# -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='main.')
    parser.add_argument('--env', type=str,help='')
    parser.add_argument('--type', type=str,help='',default='awac')
    parser.add_argument('--seed', type=str,help='',default='3')
    args = parser.parse_args()
    
    export_all_scalars_to_csv_append(f"O2ORL/log/{args.env}-human-v1/{args.type}_seed{args.seed}")
