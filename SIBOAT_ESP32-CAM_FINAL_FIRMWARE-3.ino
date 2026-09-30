/* =====================================================================
   SIBOAT  -  ESP32-CAM  AI CLASSIFIER + MJPEG STREAMER   (FINAL FIRMWARE)
   CLEANER WATERS - GREENER FUTURE
   ---------------------------------------------------------------------
   Board ........ AI Thinker ESP32-CAM   (OV3660, PSRAM enabled)
   Core ......... Arduino-ESP32 2.0.11
   Partition .... "Huge APP (3MB No OTA/1MB SPIFFS)"  <- recommended, the
                  Edge Impulse model + Wi-Fi + camera are a large image.
   Library ...... Edge Impulse Arduino library (the supplied ZIP)
                  TRI-SEABOT_CLASSIFICATION_SYSTEM_inferencing
                  (siboat_inferencing.h is used instead if you rename it)

   TASKS
     Core 1 : streamTask  - grabs QVGA JPEG frames (~15 fps), serves the
                            MJPEG stream on :81/stream, and hands every
                            5th frame to the classifier.
                            loop() (UART, Wi-Fi, heartbeat) also runs here.
     Core 0 : inferTask   - JPEG -> RGB -> 96x96 -> Edge Impulse FOMO.

   WIRING (UART to Main ESP32, 115200 8N1)
     CAM GPIO14 (TX) -> Main GPIO35 (RX)
     CAM GPIO15 (RX) <- Main GPIO17 (TX)       + common GND

   PROTOCOL
     CAM -> MAIN  <CAT,2,0.84,SEQ,18,label>   <REJ,0.84,SEQ,20,label>
                  <HB,ip,uptimeS,fps,modelOk,camOk>   <ERR,MODEL|CAM,code>
     MAIN -> CAM  ACK1 / ACK2 / ACK3 / REJECT / ERROR   and  <MHB,cls,safe,estop>
   ===================================================================== */

#if defined(__has_include)
  #if __has_include(<siboat_inferencing.h>)
    #include <siboat_inferencing.h>
  #else
    #include <siboat_inferencing.h>
  #endif
#else
  #include <siboat_inferencing.h>
#endif
#include "edge-impulse-sdk/dsp/image/image.hpp"

#include <WiFi.h>
#include <ESPmDNS.h>
#include "esp_camera.h"
#include "img_converters.h"
#include "esp_heap_caps.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include <stdarg.h>
#include <string.h>
#include <strings.h>
#include <stdlib.h>
#include "soc/soc.h"
#include "soc/rtc_cntl_reg.h"

#if !defined(EI_CLASSIFIER_SENSOR) || EI_CLASSIFIER_SENSOR != EI_CLASSIFIER_SENSOR_CAMERA
#error "The Edge Impulse library is not an image (camera) model."
#endif
#if EI_CLASSIFIER_OBJECT_DETECTION != 1
#error "This firmware expects the FOMO object-detection model."
#endif

/* ============================ NETWORK ================================ */
struct WifiCred { const char* ssid; const char* pass; };
static const WifiCred WIFI_CREDS[] = {
  { "SIBOAT.DEV", "siboatdevteam"        },   /* installation guide (primary) */
  { "SIBOAT",     "siboattrashcollector" },   /* project brief                */
  { "siboat.dev", "siboatdevteam"        }    /* camera-credentials line      */
};
#define WIFI_CRED_COUNT          (sizeof(WIFI_CREDS) / sizeof(WIFI_CREDS[0]))
#define WIFI_ATTEMPT_TIMEOUT_MS  12000UL
#define WIFI_RETRY_GAP_MS        1500UL
static IPAddress STATIC_IP(10, 90, 102, 51);        /* ESP32-CAM              */
#define MDNS_NAME                "siboat-cam"       /* -> siboat-cam.local    */
#define DEVICE_HOSTNAME          "SIBOAT-CAM"
#define STREAM_PORT              81                 /* http://<ip>:81/stream  */

/* ============================== PINS ================================= */
#define PIN_UART_TX     14
#define PIN_UART_RX     15
#define PIN_FLASH_LED   4
#define PIN_RED_LED     33          /* on-board, active LOW */

/* AI Thinker ESP32-CAM camera pins */
#define PWDN_GPIO_NUM   32
#define RESET_GPIO_NUM  -1
#define XCLK_GPIO_NUM    0
#define SIOD_GPIO_NUM   26
#define SIOC_GPIO_NUM   27
#define Y9_GPIO_NUM     35
#define Y8_GPIO_NUM     34
#define Y7_GPIO_NUM     39
#define Y6_GPIO_NUM     36
#define Y5_GPIO_NUM     21
#define Y4_GPIO_NUM     19
#define Y3_GPIO_NUM     18
#define Y2_GPIO_NUM      5
#define VSYNC_GPIO_NUM  25
#define HREF_GPIO_NUM   23
#define PCLK_GPIO_NUM   22

/* ============================ CAMERA / STREAM ======================== */
#define CAM_FRAME_SIZE      FRAMESIZE_QVGA      /* 320 x 240              */
#define CAM_W               320
#define CAM_H               240
#define CAM_JPEG_QUALITY    12
#define CAM_HMIRROR         0                   /* horizontal mirror: NO  */
#define CAM_VFLIP           0                   /* vertical flip:     NO  */
#define TARGET_FPS          15
#define STREAM_MAX_CLIENTS  2
#define INFER_JPG_CAP       (48 * 1024)
#define CAM_FAIL_REBOOT_MS  15000UL
#define DISABLE_BROWNOUT_DETECTOR 0             /* 1 = ignore weak supplies */

/* ============================ CLASSIFICATION ========================= */
#define CONF_THRESHOLD        0.50f
#define SAMPLE_EVERY_N_FRAMES 5
#define CONSEC_REQUIRED       2
#define CONSECUTIVE_MATCH_BY_CLASS 0    /* 0 = same model label, 1 = same CAT */
#define DUP_LOCKOUT_MS        2000UL
#define NEW_OBJECT_DELAY_MS   1000UL

/* ============================ UART / TIMING ========================== */
#define UART_BAUD             115200
#define ACK_TIMEOUT_MS        3000UL
#define MAX_ATTEMPTS          5
#define RETRY_COOLDOWN_MS     2000UL    /* brief = 2 s (document says 1 s) */
#define HB_PERIOD_MS          1000UL
#define MAIN_HB_TIMEOUT_MS    5000UL

/* ===================== MODEL LABEL -> MACHINE CLASS ================== */
/* cls: 1 = CAT1 plastic bottles & containers, 2 = CAT2 soft plastics,
        3 = CAT3 biodegradables, 0 = REJECT (log only)                   */
struct LabelMap { const char* label; uint8_t cls; };
static const LabelMap LABEL_MAP[] = {
  { "Face-Mask",            2 },
  { "PaperBag",             3 },
  { "Plastic-Bag",          2 },
  { "Plastic-Bottle",       1 },
  { "aluminum",             0 },
  { "cardboard",            3 },
  { "egg shell",            3 },
  { "facemask",             2 },
  { "food wrapper",         2 },
  { "fruit peels",          3 },
  { "glass bottle",         0 },
  { "left-over food",       0 },
  { "paper",                3 },
  { "pet bottle",           1 },
  { "plastic bag",          2 },
  { "plastic bottle",       1 },
  { "plastic container",    1 },
  { "plastic sachet",       2 },
  { "plastic straw",        2 },
  { "styrofoam containers", 2 },
  { "treeleaves",           3 },
  { "vegetable peels",      3 }
};
#define LABEL_MAP_COUNT (sizeof(LABEL_MAP) / sizeof(LABEL_MAP[0]))

static int classOf(const char* label) {
  if (!label) return -1;
  for (size_t i = 0; i < LABEL_MAP_COUNT; i++)
    if (strcasecmp(label, LABEL_MAP[i].label) == 0) return (int)LABEL_MAP[i].cls;
  return -1;                                    /* UNKNOWN */
}

/* ============================== STATE ================================ */
struct TxReq { uint8_t cls; float conf; char label[28]; };
enum { TX_IDLE = 0, TX_WAIT_ACK = 1, TX_COOLDOWN = 2 };

static WiFiServer       streamServer(STREAM_PORT);
static WiFiClient       sClient[STREAM_MAX_CLIENTS];
static uint8_t          sMode[STREAM_MAX_CLIENTS];     /* 0 stream, 1 snapshot */
static bool             sUsed[STREAM_MAX_CLIENTS];

static TaskHandle_t     inferTaskHandle = NULL;
static QueueHandle_t    txQueue = NULL;

static uint8_t*         inferJpg = NULL;
static uint8_t*         rgbFull  = NULL;               /* 320x240x3 */
static uint8_t*         rgbNet   = NULL;               /* 96x96x3   */
static volatile size_t  inferLen = 0;
static volatile bool    inferBusy = false;

static volatile bool    camOk = false, modelOk = true;
static volatile float   camFps = 0;
static volatile uint32_t frameCount = 0;
static volatile uint32_t inferCount = 0;
static volatile uint32_t inferMs = 0;

/* filter */
static volatile uint8_t  consecCount = 0;
static char              consecLabel[28] = "";
static int               consecClass = -2;
static volatile bool     latched = false;
static volatile uint32_t lastValidMs = 0;
static volatile uint32_t lockoutUntil = 0;
static char              lastLabel[28] = "";
static volatile float    lastConf = 0;

/* main link */
static volatile bool     mainCls = false, mainSafe = true, mainEstop = false;
static volatile uint32_t mainMhbMs = 0;
static volatile bool     txBusy = false;
static uint8_t           txState = TX_IDLE;
static TxReq             curReq;
static char              curPkt[100];
static char              curExpect[10];
static uint32_t          curSeq = 0, seqCounter = 0, txT0 = 0;
static uint8_t           attempts = 0;
static uint32_t          txOk = 0, txFail = 0, txRetries = 0;
static char              uLine[64];
static uint8_t           uLen = 0;
static uint32_t          lastHb = 0, bootMs = 0;
static volatile bool     errPendingModel = false, errPendingCam = false;

/* wifi */
static uint8_t  wifiIdx = 0, staticTries = 0;
static bool     wifiTrying = false, wifiStaticAttempt = false, haveNet = false, servicesUp = false;
static uint32_t wifiT0 = 0, wifiLastEnd = 0;
static IPAddress netGw, netMask;

static bool classificationAllowed() {
  return mainMhbMs != 0 && (millis() - mainMhbMs) < MAIN_HB_TIMEOUT_MS && mainCls && !mainSafe && !mainEstop;
}
static void* allocBuf(size_t n) {
  void* p = heap_caps_malloc(n, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  if (!p) p = malloc(n);
  return p;
}
static void copyStr(char* dst, size_t n, const char* src) {
  strncpy(dst, src ? src : "", n - 1);
  dst[n - 1] = 0;
}

/* ============================== CAMERA =============================== */
static bool cameraInit() {
  camera_config_t config;
  memset(&config, 0, sizeof(config));
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size = CAM_FRAME_SIZE;
  config.jpeg_quality = CAM_JPEG_QUALITY;
  config.fb_count = 2;
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.grab_mode = CAMERA_GRAB_LATEST;

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("[CAM] init failed 0x%x\n", (unsigned)err);
    return false;
  }
  sensor_t* s = esp_camera_sensor_get();
  if (s) {
    Serial.printf("[CAM] sensor PID 0x%04x %s\n", (unsigned)s->id.PID, s->id.PID == OV3660_PID ? "(OV3660)" : "");
    s->set_framesize(s, CAM_FRAME_SIZE);
    s->set_quality(s, CAM_JPEG_QUALITY);
    s->set_hmirror(s, CAM_HMIRROR);
    s->set_vflip(s, CAM_VFLIP);
    if (s->id.PID == OV3660_PID) {
      s->set_brightness(s, 1);
      s->set_saturation(s, 0);
    }
  }
  return true;
}

/* ============================ MJPEG SERVER =========================== */
static const char CORS_HDR[] =
  "Access-Control-Allow-Origin: *\r\n"
  "Access-Control-Allow-Private-Network: true\r\n"
  "Access-Control-Allow-Methods: GET, OPTIONS\r\n"
  "Cache-Control: no-cache, no-store, must-revalidate\r\n";

static void httpSimple(WiFiClient& c, int code, const char* type, const char* body) {
  char h[256];
  int n = snprintf(h, sizeof(h), "HTTP/1.1 %d %s\r\nContent-Type: %s\r\nContent-Length: %u\r\n%sConnection: close\r\n\r\n",
                   code, code == 200 ? "OK" : (code == 204 ? "No Content" : "Not Found"), type,
                   (unsigned)strlen(body), CORS_HDR);
  c.write((const uint8_t*)h, n);
  if (*body) c.write((const uint8_t*)body, strlen(body));
}
static void buildStatusJson(char* out, size_t n) {
  IPAddress ip = WiFi.localIP();
  int clients = 0;
  for (int i = 0; i < STREAM_MAX_CLIENTS; i++) if (sUsed[i] && sMode[i] == 0) clients++;
  snprintf(out, n,
           "{\"ok\":true,\"ip\":\"%d.%d.%d.%d\",\"fps\":%.1f,\"frames\":%lu,\"camOk\":%s,\"modelOk\":%s,"
           "\"classificationAllowed\":%s,\"inferences\":%lu,\"inferMs\":%lu,\"lastLabel\":\"%s\","
           "\"lastConfidence\":%.2f,\"txOk\":%lu,\"txFail\":%lu,\"streamClients\":%d,\"uptime\":%lu}",
           ip[0], ip[1], ip[2], ip[3], (double)camFps, (unsigned long)frameCount,
           camOk ? "true" : "false", modelOk ? "true" : "false",
           classificationAllowed() ? "true" : "false", (unsigned long)inferCount, (unsigned long)inferMs,
           lastLabel, (double)lastConf, (unsigned long)txOk, (unsigned long)txFail, clients,
           (unsigned long)(millis() / 1000UL));
}
static void acceptClients() {
  WiFiClient nc = streamServer.available();
  if (!nc) return;
  char req[200];
  size_t rl = 0;
  uint32_t t0 = millis();
  while (millis() - t0 < 400 && rl < sizeof(req) - 1) {
    if (nc.available()) {
      req[rl++] = (char)nc.read();
      if (rl >= 4 && memcmp(req + rl - 4, "\r\n\r\n", 4) == 0) break;
    } else if (!nc.connected()) {
      break;
    } else {
      vTaskDelay(1);
    }
  }
  req[rl] = 0;
  nc.setNoDelay(true);

  bool isGet = strncmp(req, "GET ", 4) == 0;
  bool isOpt = strncmp(req, "OPTIONS ", 8) == 0;
  const char* path = isGet ? req + 4 : (isOpt ? req + 8 : "");
  bool pStream  = isGet && strncmp(path, "/stream", 7) == 0;
  bool pCapture = isGet && (strncmp(path, "/capture", 8) == 0 || strncmp(path, "/snapshot", 9) == 0);
  bool pStatus  = isGet && strncmp(path, "/status", 7) == 0;
  bool pRoot    = isGet && (path[0] == '/' && (path[1] == ' ' || path[1] == '?'));

  if (isOpt)       { httpSimple(nc, 204, "text/plain", ""); nc.stop(); return; }
  if (pStatus)     { char j[420]; buildStatusJson(j, sizeof(j)); httpSimple(nc, 200, "application/json", j); nc.stop(); return; }
  if (pRoot)       { httpSimple(nc, 200, "text/plain", "SIBOAT ESP32-CAM online.\n/stream  MJPEG\n/capture  JPEG\n/status   JSON\n"); nc.stop(); return; }
  if (!pStream && !pCapture) { httpSimple(nc, 404, "text/plain", "not found"); nc.stop(); return; }

  int slot = -1;
  for (int i = 0; i < STREAM_MAX_CLIENTS; i++) if (!sUsed[i] || !sClient[i].connected()) { slot = i; break; }
  if (slot < 0) { httpSimple(nc, 404, "text/plain", "busy"); nc.stop(); return; }
  if (sUsed[slot]) sClient[slot].stop();
  nc.setTimeout(2);
  sClient[slot] = nc;
  sUsed[slot] = true;
  sMode[slot] = pCapture ? 1 : 0;
  if (pStream) {
    char h[256];
    int n = snprintf(h, sizeof(h),
                     "HTTP/1.1 200 OK\r\nContent-Type: multipart/x-mixed-replace;boundary=frame\r\n%sConnection: close\r\n\r\n",
                     CORS_HDR);
    sClient[slot].write((const uint8_t*)h, n);
    Serial.printf("[STREAM] viewer connected (slot %d)\n", slot);
  }
}
static void dropClient(int i) {
  sClient[i].stop();
  sUsed[i] = false;
}
static void serveClients(camera_fb_t* fb) {
  for (int i = 0; i < STREAM_MAX_CLIENTS; i++) {
    if (!sUsed[i]) continue;
    WiFiClient& c = sClient[i];
    if (!c.connected()) { dropClient(i); Serial.println("[STREAM] viewer left"); continue; }
    if (sMode[i] == 1) {
      char h[256];
      int n = snprintf(h, sizeof(h), "HTTP/1.1 200 OK\r\nContent-Type: image/jpeg\r\nContent-Length: %u\r\n%sConnection: close\r\n\r\n",
                       (unsigned)fb->len, CORS_HDR);
      c.write((const uint8_t*)h, n);
      c.write(fb->buf, fb->len);
      dropClient(i);
      continue;
    }
    char h[96];
    int n = snprintf(h, sizeof(h), "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n", (unsigned)fb->len);
    if (c.write((const uint8_t*)h, n) != (size_t)n ||
        c.write(fb->buf, fb->len) != fb->len ||
        c.write((const uint8_t*)"\r\n", 2) != 2) {
      dropClient(i);
      Serial.println("[STREAM] viewer dropped (write failed)");
    }
  }
}

/* ============================ STREAM TASK (core 1) =================== */
static void streamTask(void* arg) {
  const uint32_t frameGap = 1000UL / TARGET_FPS;
  uint32_t nextFrame = millis();
  uint32_t fpsT0 = millis(), fpsCount = 0, badSince = 0;
  bool prevAllowed = false;
  for (;;) {
    if (servicesUp) acceptClients();

    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb) {
      if (!badSince) badSince = millis();
      if (millis() - badSince > 3000UL) { if (camOk) errPendingCam = true; camOk = false; }
      if (millis() - badSince > CAM_FAIL_REBOOT_MS) { Serial.println("[CAM] no frames - rebooting"); delay(50); ESP.restart(); }
      vTaskDelay(pdMS_TO_TICKS(40));
      continue;
    }
    badSince = 0;
    camOk = true;
    frameCount++;
    fpsCount++;

    bool allowed = classificationAllowed();
    if (allowed && !prevAllowed) { latched = false; consecCount = 0; consecLabel[0] = 0; }
    prevAllowed = allowed;

    if (allowed && (frameCount % SAMPLE_EVERY_N_FRAMES) == 0 && !inferBusy &&
        fb->format == PIXFORMAT_JPEG && fb->len <= INFER_JPG_CAP && fb->width == CAM_W && fb->height == CAM_H) {
      memcpy(inferJpg, fb->buf, fb->len);
      inferLen = fb->len;
      inferBusy = true;
      if (inferTaskHandle) xTaskNotifyGive(inferTaskHandle);
    }

    serveClients(fb);
    esp_camera_fb_return(fb);

    uint32_t now = millis();
    if (now - fpsT0 >= 1000UL) {
      camFps = (float)fpsCount * 1000.0f / (float)(now - fpsT0);
      fpsCount = 0;
      fpsT0 = now;
    }
    nextFrame += frameGap;
    int32_t wait = (int32_t)(nextFrame - millis());
    if (wait > 0) vTaskDelay(pdMS_TO_TICKS(wait));
    else { nextFrame = millis(); vTaskDelay(1); }
  }
}

/* ========================== INFERENCE TASK (core 0) ================== */
static int camGetData(size_t offset, size_t length, float* out_ptr) {
  size_t ix = offset * 3;
  for (size_t i = 0; i < length; i++) {
    /* fmt2rgb888 yields BGR order (esp32-camera issue #379): swap to 0xRRGGBB */
    out_ptr[i] = (float)((rgbNet[ix + 2] << 16) + (rgbNet[ix + 1] << 8) + rgbNet[ix]);
    ix += 3;
  }
  return 0;
}
static void queueTx(uint8_t cls, float conf, const char* label) {
  TxReq r;
  r.cls = cls;
  r.conf = conf;
  copyStr(r.label, sizeof(r.label), label);
  txBusy = true;
  if (xQueueSend(txQueue, &r, 0) != pdTRUE) txBusy = false;
}
static void handleDetection(bool have, const char* label, float conf, int cls) {
  uint32_t now = millis();
  if (!have) {
    consecCount = 0;
    consecLabel[0] = 0;
    if (latched && (now - lastValidMs) >= NEW_OBJECT_DELAY_MS) {
      latched = false;
      Serial.println("[AI] object gone - ready for a new object");
    }
    return;
  }
  lastValidMs = now;
  copyStr(lastLabel, sizeof(lastLabel), label);
  lastConf = conf;
  if (latched) return;                               /* same object still in view */
  if ((int32_t)(lockoutUntil - now) > 0) { consecCount = 0; return; }
  if (txBusy) return;

#if CONSECUTIVE_MATCH_BY_CLASS
  bool same = (consecCount > 0 && cls == consecClass);
#else
  bool same = (consecCount > 0 && strcasecmp(label, consecLabel) == 0);
#endif
  if (same) consecCount++;
  else { consecCount = 1; copyStr(consecLabel, sizeof(consecLabel), label); consecClass = cls; }
  Serial.printf("[AI] valid %s (%.2f) -> %s  consecutive %u/%u\n", label, (double)conf,
                cls == 0 ? "REJECT" : (cls == 1 ? "CAT1" : (cls == 2 ? "CAT2" : "CAT3")),
                (unsigned)consecCount, (unsigned)CONSEC_REQUIRED);
  if (consecCount >= CONSEC_REQUIRED) {
    consecCount = 0;
    latched = true;
    queueTx((uint8_t)cls, conf, label);
  }
}
static void inferTask(void* arg) {
  static ei_impulse_result_t result;
  uint8_t errRun = 0;
  uint32_t quiet = 0;
  for (;;) {
    ulTaskNotifyTake(pdTRUE, pdMS_TO_TICKS(1000));
    if (!inferBusy) continue;

    uint32_t t0 = millis();
    bool ok = fmt2rgb888(inferJpg, inferLen, PIXFORMAT_JPEG, rgbFull);
    if (!ok) {
      if (++errRun >= 3 && modelOk) { modelOk = false; errPendingModel = true; }
      inferBusy = false;
      continue;
    }
    ei::image::processing::crop_and_interpolate_rgb888(rgbFull, CAM_W, CAM_H, rgbNet,
                                                       EI_CLASSIFIER_INPUT_WIDTH, EI_CLASSIFIER_INPUT_HEIGHT);
    ei::signal_t signal;
    signal.total_length = EI_CLASSIFIER_INPUT_WIDTH * EI_CLASSIFIER_INPUT_HEIGHT;
    signal.get_data = &camGetData;

    EI_IMPULSE_ERROR err = run_classifier(&signal, &result, false);
    inferMs = millis() - t0;
    inferCount++;
    if (err != EI_IMPULSE_OK) {
      Serial.printf("[AI] run_classifier error %d\n", (int)err);
      if (++errRun >= 3 && modelOk) { modelOk = false; errPendingModel = true; }
      inferBusy = false;
      continue;
    }
    errRun = 0;
    modelOk = true;

    /* highest-confidence valid object above threshold */
    const char* bestLabel = NULL;
    float bestConf = 0;
    int bestCls = -1;
    for (uint32_t i = 0; i < result.bounding_boxes_count; i++) {
      ei_impulse_result_bounding_box_t bb = result.bounding_boxes[i];
      if (bb.value == 0 || bb.value < CONF_THRESHOLD) continue;
      int c = classOf(bb.label);
      if (c < 0) continue;                           /* UNKNOWN: ignored, never transmitted */
      if (bb.value > bestConf) { bestConf = bb.value; bestLabel = bb.label; bestCls = c; }
    }
    handleDetection(bestLabel != NULL, bestLabel ? bestLabel : "", bestConf, bestCls);
    if (!bestLabel && (++quiet % 25) == 0) Serial.printf("[AI] idle  (%lu ms/inference)\n", (unsigned long)inferMs);
    inferBusy = false;
  }
}

/* ============================ UART LINK ============================== */
static void uartSend(const char* pkt) { Serial1.print(pkt); }

static void txSendAttempt() {
  attempts++;
  if (attempts > 1) txRetries++;
  uartSend(curPkt);
  txState = TX_WAIT_ACK;
  txT0 = millis();
  Serial.printf("[UART] TX #%u/%u  %s\n", (unsigned)attempts, (unsigned)MAX_ATTEMPTS, curPkt);
}
static void txFinish(bool okFlag) {
  txState = TX_IDLE;
  lockoutUntil = millis() + DUP_LOCKOUT_MS;
  if (okFlag) txOk++;
  else { txFail++; latched = false; Serial.println("[UART] gave up after max attempts - object may be re-sent after lockout"); }
  txBusy = false;
}
static void txFailedAttempt(const char* why) {
  Serial.printf("[UART] attempt %u failed (%s)\n", (unsigned)attempts, why);
  if (attempts >= MAX_ATTEMPTS) { txFinish(false); return; }
  txState = TX_COOLDOWN;
  txT0 = millis();
}
static void txStart(const TxReq& r) {
  curReq = r;
  curSeq = ++seqCounter;
  attempts = 0;
  if (r.cls == 0) {
    snprintf(curPkt, sizeof(curPkt), "<REJ,%.2f,SEQ,%lu,%s>", (double)r.conf, (unsigned long)curSeq, r.label);
    copyStr(curExpect, sizeof(curExpect), "REJECT");
  } else {
    snprintf(curPkt, sizeof(curPkt), "<CAT,%u,%.2f,SEQ,%lu,%s>", (unsigned)r.cls, (double)r.conf, (unsigned long)curSeq, r.label);
    snprintf(curExpect, sizeof(curExpect), "ACK%u", (unsigned)r.cls);
  }
  txSendAttempt();
}
static void txTick() {
  uint32_t now = millis();
  if (txState == TX_IDLE) {
    TxReq r;
    if (xQueueReceive(txQueue, &r, 0) == pdTRUE) {
      if (classificationAllowed()) txStart(r);
      else { txBusy = false; latched = false; }
    }
    return;
  }
  if (!classificationAllowed()) {                    /* Main disabled us: stop retrying */
    Serial.println("[UART] classification disabled by Main - transmission aborted");
    txState = TX_IDLE; txBusy = false; latched = false;
    return;
  }
  if (txState == TX_WAIT_ACK && (now - txT0) >= ACK_TIMEOUT_MS) txFailedAttempt("timeout");
  else if (txState == TX_COOLDOWN && (now - txT0) >= RETRY_COOLDOWN_MS) txSendAttempt();
}
static void handleMhb(char* body) {                 /* body: "MHB,cls,safe,estop" */
  char* f[4];
  int n = 0;
  char* p = body;
  f[n++] = p;
  while (*p && n < 4) { if (*p == ',') { *p = 0; f[n++] = p + 1; } p++; }
  if (n < 4 || strcmp(f[0], "MHB") != 0) return;
  mainCls = (f[1][0] == '1');
  mainSafe = (f[2][0] == '1');
  mainEstop = (f[3][0] == '1');
  mainMhbMs = millis();
}
static void handleLine(char* line) {
  if (line[0] == '<') {
    size_t l = strlen(line);
    if (l > 2 && line[l - 1] == '>') { line[l - 1] = 0; handleMhb(line + 1); }
    return;
  }
  if (txState == TX_WAIT_ACK) {
    if (strcmp(line, curExpect) == 0) {
      Serial.printf("[UART] %s received after %u attempt(s)\n", line, (unsigned)attempts);
      txFinish(true);
    } else if (strcmp(line, "ERROR") == 0) {
      txFailedAttempt("Main replied ERROR");
    }
  }
}
static void uartPoll() {
  while (Serial1.available() > 0) {
    char c = (char)Serial1.read();
    if (c == '\r') continue;
    if (c == '\n') { uLine[uLen] = 0; if (uLen) handleLine(uLine); uLen = 0; }
    else if (uLen < sizeof(uLine) - 1) uLine[uLen++] = c;
    else uLen = 0;
  }
}
static void heartbeatTick() {
  uint32_t now = millis();
  if (now - lastHb < HB_PERIOD_MS) return;
  lastHb = now;
  char p[96];
  IPAddress ip = WiFi.localIP();
  snprintf(p, sizeof(p), "<HB,%d.%d.%d.%d,%lu,%.1f,%d,%d>", ip[0], ip[1], ip[2], ip[3],
           (unsigned long)(now / 1000UL), (double)camFps, modelOk ? 1 : 0, camOk ? 1 : 0);
  uartSend(p);
  if (errPendingModel) { errPendingModel = false; uartSend("<ERR,MODEL,1>"); }
  if (errPendingCam)   { errPendingCam = false;   uartSend("<ERR,CAM,1>"); }
  digitalWrite(PIN_RED_LED, (modelOk && camOk) ? ((now / 1000UL) & 1 ? HIGH : LOW) : LOW);
}

/* ============================== WI-FI ================================ */
static void wifiBegin(bool useStatic) {
  WiFi.disconnect(false, false);
  if (useStatic) WiFi.config(STATIC_IP, netGw, netMask, netGw);
  else           WiFi.config(IPAddress(0, 0, 0, 0), IPAddress(0, 0, 0, 0), IPAddress(0, 0, 0, 0));
  const WifiCred& c = WIFI_CREDS[wifiIdx % WIFI_CRED_COUNT];
  Serial.printf("[WIFI] connecting to \"%s\" (%s)...\n", c.ssid, useStatic ? "static" : "DHCP");
  WiFi.begin(c.ssid, c.pass);
  wifiTrying = true;
  wifiStaticAttempt = useStatic;
  wifiT0 = millis();
}
static void servicesStart() {
  if (!servicesUp) { streamServer.begin(); servicesUp = true; }
  MDNS.end();
  if (MDNS.begin(MDNS_NAME)) MDNS.addService("http", "tcp", STREAM_PORT);
}
static void wifiConnected() {
  wifiTrying = false;
  IPAddress ip = WiFi.localIP();
  bool inTarget = (ip[0] == STATIC_IP[0] && ip[1] == STATIC_IP[1] && ip[2] == STATIC_IP[2]);
  if (inTarget && ip != STATIC_IP && staticTries < 2) {
    staticTries++;
    netGw = WiFi.gatewayIP();
    netMask = WiFi.subnetMask();
    Serial.printf("[WIFI] DHCP gave %s - switching to preferred static %s\n", ip.toString().c_str(), STATIC_IP.toString().c_str());
    wifiBegin(true);
    return;
  }
  if (!inTarget) Serial.printf("[WIFI] hotspot subnet is %d.%d.%d.x, not 10.90.102.x - staying on DHCP.\n", ip[0], ip[1], ip[2]);
  haveNet = true;
  Serial.printf("[WIFI] connected  SSID=%s  IP=%s  RSSI=%d dBm\n", WiFi.SSID().c_str(), ip.toString().c_str(), WiFi.RSSI());
  servicesStart();
  Serial.printf("[STREAM] http://%s:%d/stream\n", ip.toString().c_str(), STREAM_PORT);
}
static void wifiTick() {
  uint32_t now = millis();
  bool settling = wifiTrying && (now - wifiT0) < 700UL;
  if (!settling && WiFi.status() == WL_CONNECTED && WiFi.localIP() != IPAddress(0, 0, 0, 0)) {
    if (wifiTrying || !haveNet) wifiConnected();
    return;
  }
  if (settling) return;
  if (haveNet) { haveNet = false; Serial.println("[WIFI] link lost - recovering"); }
  if (wifiTrying) {
    if (now - wifiT0 > WIFI_ATTEMPT_TIMEOUT_MS) {
      Serial.println("[WIFI] attempt timed out");
      wifiTrying = false;
      wifiLastEnd = now;
      if (wifiStaticAttempt) staticTries = 2;
      wifiIdx++;
      WiFi.disconnect(false, false);
    }
    return;
  }
  if (now - wifiLastEnd < WIFI_RETRY_GAP_MS) return;
  bool useStatic = (staticTries >= 1 && staticTries < 2 && netGw != IPAddress(0, 0, 0, 0));
  wifiBegin(useStatic);
}

/* =============================== SETUP =============================== */
void setup() {
#if DISABLE_BROWNOUT_DETECTOR
  WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);
#endif
  bootMs = millis();
  pinMode(PIN_FLASH_LED, OUTPUT);
  digitalWrite(PIN_FLASH_LED, LOW);
  pinMode(PIN_RED_LED, OUTPUT);
  digitalWrite(PIN_RED_LED, HIGH);
  Serial.begin(115200);
  delay(100);
  Serial.println();
  Serial.println("=== SIBOAT ESP32-CAM - FINAL FIRMWARE ===");
  Serial.printf("[MODEL] %dx%d, %d labels, threshold %.2f, every %dth frame\n",
                EI_CLASSIFIER_INPUT_WIDTH, EI_CLASSIFIER_INPUT_HEIGHT, EI_CLASSIFIER_LABEL_COUNT,
                (double)CONF_THRESHOLD, SAMPLE_EVERY_N_FRAMES);

  Serial1.setRxBufferSize(256);
  Serial1.begin(UART_BAUD, SERIAL_8N1, PIN_UART_RX, PIN_UART_TX);

  bool psram = psramFound();
  Serial.printf("[MEM] PSRAM %s, free heap %u, free PSRAM %u\n", psram ? "found" : "MISSING",
                (unsigned)ESP.getFreeHeap(), (unsigned)ESP.getFreePsram());
  inferJpg = (uint8_t*)allocBuf(INFER_JPG_CAP);
  rgbFull  = (uint8_t*)allocBuf((size_t)CAM_W * CAM_H * 3);
  rgbNet   = (uint8_t*)allocBuf((size_t)EI_CLASSIFIER_INPUT_WIDTH * EI_CLASSIFIER_INPUT_HEIGHT * 3);
  bool memOk = psram && inferJpg && rgbFull && rgbNet;
  if (!memOk) { Serial.println("[MEM] buffer allocation FAILED"); modelOk = false; errPendingModel = true; }

  txQueue = xQueueCreate(1, sizeof(TxReq));

  camOk = cameraInit();
  if (!camOk) errPendingCam = true;

  if (camOk && memOk) {
    xTaskCreatePinnedToCore(inferTask, "infer", 16384, NULL, 2, &inferTaskHandle, 0);
  }
  if (camOk) {
    xTaskCreatePinnedToCore(streamTask, "stream", 8192, NULL, 3, NULL, 1);
  }

  WiFi.persistent(false);
  WiFi.setHostname(DEVICE_HOSTNAME);
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  WiFi.setAutoReconnect(true);
  wifiBegin(false);
}

/* ================================ LOOP =============================== */
void loop() {
  wifiTick();
  uartPoll();
  txTick();
  heartbeatTick();
  if (!camOk && (millis() - bootMs) > 20000UL) { Serial.println("[CAM] camera never started - rebooting"); delay(50); ESP.restart(); }
  delay(2);
}
