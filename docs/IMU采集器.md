# IMU + GPS 数据采集器

Android App，采集加速度计 + 陀螺仪 + GPS 数据，用于训练惯性导航模型。

## 编译

```bash
cd ~/桌面/IMUCollector/IMUCollector
./gradlew assembleDebug
adb install app/build/outputs/apk/debug/app-debug.apk
```

## 数据格式

CSV:
```
timestamp,ax,ay,az,gx,gy,gz,lat,lng,alt,speed,bearing
秒,加速度(m/s²),陀螺仪(rad/s),GPS纬度,经度,高度,速度,方向
```

## 采集步骤

1. 打开 App → 点「开始」
2. 正常走路/开车
3. 点「停止」
4. 数据文件: App 内部 Documents/IMUData/
