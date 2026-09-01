# websocket_xyzq_stream.py 使用流程

## 功能概要

`websocket_xyzq_stream.py` 是 PC + Raspberry Pi 的即時人體標定流程。

流程：

1. PC 開 WebSocket server。
2. Pi 連到 PC。
3. 兩邊相機 ready 後，PC 倒數並同步開始錄影。
4. PC/Pi 各自錄影與跑 MediaPipe pose。
5. Pi 即時傳 pose JSON；若開啟 IMU 融合，Pi 會同時直接從序列埠讀 IMU frame 並送到 PC。
6. PC 即時配對 cam0/cam2 pose，計算人體 xyzq 與 reprojection error。
7. 若有開 IMU 融合，PC 會把 IMU 當成即時輔助，先做 rotation 融合，translation 先維持視覺結果。
8. 錄完後 Pi 傳回 raw mp4 和 pose mp4。
9. 固定輸出 summary 表格，並可選擇自動跑 3D 影片、ArUco 外參、棋盤外參。

相機與檔案對應：

```text
cam0 = PC camera
cam2 = Pi camera
--k1 = data/K0.txt    PC cam0 內參
--k2 = data/K1.txt    Pi cam2 內參
--d1 = data/dist0.txt PC cam0 畸變
--d2 = data/dist1.txt Pi cam2 畸變
```

## 基本跑法

### 1. PC 端先跑

```powershell
cd C:\Users\70929\Desktop\new

.\.venv\Scripts\python.exe .\websocket_xyzq_stream.py server `
  --host 0.0.0.0 `
  --port 8765 `
  --tag test_1 `
  --pc-cam 0 `
  --fps 10 `
  --secs 10 `
  --start-delay 3 `
  --solve-batch-size 1 `
  --smooth-window 5 `
  --vis 0.3 `
  --min-pairs 6 `
  --k1 .\data\K0.txt `
  --k2 .\data\K1.txt `
  --d1 .\data\dist0.txt `
  --d2 .\data\dist1.txt
```

看到這行代表 PC 正在等 Pi，不是卡住：

```text
[server] waiting on ws://0.0.0.0:8765
```

### 2. Pi 端再跑

```bash
cd ~/pose_code
~/pose_env/bin/python websocket_xyzq_stream.py pi \
  --server ws://192.168.1.104:8765 \
  --pi-cam 0
```

把 `192.168.1.104` 換成 PC 的 IP。

## 常用參數

```text
--secs 10              錄 10 秒；設 0 則手動 Enter 停止
--start-delay 3        ready 後倒數 3 秒再開始
--solve-batch-size 1   一筆 matched pose 算一筆 xyzq
--solve-batch-size 5   五筆 matched pose 合併算一筆 xyzq
--smooth-window 5      xyzq 平滑視窗
--smooth-window 1      關閉 xyzq 平滑
--vis 0.3              pose visibility 門檻
--min-pairs 6          至少 6 個共同人體點才算 xyzq
--enable-imu-fusion    預設關閉；開啟後 Pi 端會直接讀 IMU 並啟用即時 rotation 融合
```

如果要更接近原本離線流程：

```powershell
--vis 0.5 --min-pairs 8 --solve-batch-size 1 --smooth-window 1
```

## 即時輸出

成功算出 xyzq 時，PC 端會印：

```text
[xyzq] ts=1.200 batch=1 pts=9 x=... y=... z=... q=(...) reproj=4.32px cam0=3.91px cam2=4.69px
```

其中：

```text
reproj = cam0/cam2 合併 reprojection RMSE
cam0   = PC cam0 reprojection RMSE
cam2   = Pi cam2 reprojection RMSE
```

### IMU 在螢幕上的行為

目前 IMU 是背景執行，不會每一筆都直接印在螢幕上。

當 `--enable-imu-fusion` 開啟時：

- Pi 端會啟動一個背景 thread，直接從 IMU 序列埠讀 frame
- 會套用和 `pose_fin.py` 相同的 frame 解析、範圍檢查與座標 remap
- 解析後的 IMU sample 會即時送到 PC
- PC 端會把 IMU sample 放進 buffer，供後續 xyzq 的 rotation 融合使用

所以目前螢幕上主要仍是：

- PC 的 `xyzq` 輸出
- 一般錄影 / 串流狀態訊息

如果 IMU 裝置有問題，會以錯誤的方式反映到流程中，而不是默默假裝沒事。

## 後處理選項

### 3D 影片與 HTML

```powershell
--make-3d-video
```

用途：把人體 xyzq 和 matched joints 做 3D 重建，方便看骨架、相機位置與跳動狀況。

輸出：

```text
one_program_test/output/<tag>_3d.mp4
one_program_test/output/<tag>_3d.html
```

### ArUco 外參

```powershell
--run-aruco
```

用途：用 `aruco3.py` 算 ArUco 參考外參，和人體標定比較。

固定參數：

```text
marker_size = 0.194
dict = DICT_6X6_50
stride = 1
smooth = 5
```

輸出：

```text
one_program_test/data/rt_aruco_<tag>.txt
```

### 棋盤外參

```powershell
--run-chessboard
```

用途：用 `chessboard_stereo_calibrate.py` 預設參數算棋盤參考外參，和人體標定比較。

輸出：

```text
one_program_test/data/rt_chessboard_<tag>.txt
```

### Summary 表格

summary 每次都會輸出，不需要額外加參數。

輸出：

```text
one_program_test/results/summary.csv
```

內容包含：

```text
tag、測試時長、xyzq 筆數
人體 cam0/cam2/combined reprojection min/max/mean
ArUco cam0/cam2/combined reprojection min/max/mean
棋盤 cam0/cam2/combined reprojection min/max/mean
人體 vs ArUco 的 normalized translation error 與 rotation error
人體 vs 棋盤的 normalized translation error 與 rotation error
```

### 全部後處理

```powershell
--postprocess-all
```

等同於：

```text
--make-3d-video --run-aruco --run-chessboard
```

summary 本來就會固定輸出。

完整範例：

```powershell
.\.venv\Scripts\python.exe .\websocket_xyzq_stream.py server `
  --host 0.0.0.0 `
  --port 8765 `
  --tag test_full `
  --pc-cam 0 `
  --fps 10 `
  --secs 10 `
  --start-delay 3 `
  --solve-batch-size 1 `
  --smooth-window 5 `
  --vis 0.3 `
  --min-pairs 6 `
  --k1 .\data\K0.txt `
  --k2 .\data\K1.txt `
  --d1 .\data\dist0.txt `
  --d2 .\data\dist1.txt `
  --postprocess-all
```

Pi 端跑法不變。

## 輸出位置

所有 PC 端輸出集中在：

```text
one_program_test/
```

重要檔案：

```text
one_program_test/data/xyzq_<tag>.jsonl
one_program_test/data/cam_match_<tag>.jsonl
one_program_test/data/rt_aruco_<tag>.txt
one_program_test/data/rt_chessboard_<tag>.txt
one_program_test/results/summary.csv
one_program_test/output/<tag>_3d.mp4
one_program_test/output/<tag>_3d.html
one_program_test/pc_pose_video/raw/cam0_<tag>.mp4
one_program_test/pc_pose_video/video/cam0_<tag>.mp4
one_program_test/pi_pose_video/raw/cam2_<tag>.mp4
one_program_test/pi_pose_video/video/cam2_<tag>.mp4
```

## Video file naming update

`websocket_xyzq_stream.py` now keeps both video files for each camera:

```text
one_program_test/pc_pose_video/raw/cam0_<session_tag>.mp4
one_program_test/pc_pose_video/video/cam0_<session_tag>.mp4
one_program_test/pi_pose_video/raw/cam2_<session_tag>.mp4
one_program_test/pi_pose_video/video/cam2_<session_tag>.mp4
```

`<session_tag>` always includes the current timestamp.

Examples:

```text
cam0_20260814_153012.mp4
cam2_20260814_153012.mp4
```

If you pass `--tag test_1`, the real session tag becomes:

```text
test_1_20260814_153012
```

This avoids overwriting older recordings while still keeping the user label.

## 注意事項

人體 xyzq 的 translation 來自 `recoverPose`，沒有真實尺度；和 ArUco/棋盤比較 translation 時，程式使用 normalize 後的方向誤差。

IMU 目前只作為即時輔助：先修 rotation，不直接拿來做 translation/尺度。

IMU 融合預設是關閉的；只有加上 `--enable-imu-fusion` 才會啟用 Pi 端序列埠讀取與 PC 端 rotation 融合。

如果 `xyzq_count` 很低或為 0，先試：

```powershell
--vis 0.3 --min-pairs 6
```

如果 Pi 連不上，檢查：

```text
PC IP 是否正確
Windows 防火牆是否擋 8765
PC 和 Pi 是否在同一網路
Pi camera index 是否正確
```
