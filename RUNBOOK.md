# 실시간 모드 운영 실행 순서

모든 프로그램을 종료한 상태에서 처음부터 다시 시작할 때 쓰는 순서. 학교 서버 혼잡도 웹으로 실시간 결과를 보낸다 (spec 9.4).

```
CCTV ─RTSP─▶ YOBA (노트북, 모델 실행) ══SSH 터널══▶ 학교 서버 Redis ─▶ 웹 서버 ─▶ 방문객
```

모델은 노트북에서만 돈다. 서버로는 결과(사람 수 JSON, 블러 JPEG)만 간다.

이 저장소는 공개라서 접속 정보는 자리표시로 적었다. 실제 값은 웹 담당자의 연결 안내문에 있다.

| 자리표시 | 뜻 |
|---|---|
| `<SSH_주소>` | 학교 서버 SSH 계정@주소 |
| `<SSH_포트>` | 학교 서버 SSH 포트 |
| `<CCTV_IP>` | CCTV IP 주소 |

## 0. 한 번만 준비

- `.env`에 Redis 주소를 넣는다. git에서 제외되므로 한 번 넣으면 계속 쓴다.
  ```
  YOBA_REDIS_URL=redis://:레디스비밀번호@localhost:6379/0
  ```
  비밀번호 앞의 `:`는 지우지 않는다.
- CCTV 유선 케이블을 노트북에 꽂는다.
- (권장) 터널 자동 재연결: `brew install autossh`

명령은 macOS 기본 셸(zsh) 기준이다. `read "u?..."`는 bash에서는 동작하지 않는다.

## 1. 터미널 ①: SSH 터널 (계속 켜 둔다)

```bash
ssh -o ServerAliveInterval=30 -L 6379:127.0.0.1:6379 <SSH_주소> -p <SSH_포트>
```

SSH 비밀번호를 입력하면 서버에 로그인된다. 창을 닫으면 터널이 끊기고 웹이 "확인 불가"가 된다. `ServerAliveInterval=30`은 가만히 둬도 접속이 끊기지 않게 한다.

autossh를 쓰면 끊겨도 자동으로 다시 붙는다.

```bash
autossh -M 0 -N -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -L 6379:127.0.0.1:6379 <SSH_주소> -p <SSH_포트>
```

## 2. 터미널 ②: YOBA 실행

```bash
cd ~/YOBA && read "u?CCTV 계정: " && read -s "p?CCTV 비번: " && echo && p=$(python3 -c 'import urllib.parse,sys;print(urllib.parse.quote(sys.argv[1],safe=""))' "$p") && uv run --env-file .env yoba --rtsp "rtsp://$u:$p@<CCTV_IP>:554/stream1" --live --publish
```

- CCTV 계정과 비밀번호를 물으면 입력한다. 비밀번호는 화면에 보이지 않고, 특수문자는 자동으로 URL 인코딩된다.
- `--env-file .env`를 빼면 비밀번호 없이 접속해서 Redis 전송이 실패한다.
- TP-Link 카메라는 `stream1`이 메인 스트림(1280×720), `stream2`가 서브 스트림(736×416)이다 (2026-11-12 카메라 설정 기준).
- 블러 처리된 화면 창이 뜨면 동작하는 것이다. 창에서 `q` 또는 Esc로 종료한다.

## 3. 확인

https://yoba.https.gsmsv.site/ 에서 사람 수와 화면이 움직이면 성공이다. YOBA를 끄면 3초 안에 "확인 불가"로 바뀌는 게 정상이다.

안 되면 새 터미널에서 터널과 Redis 상태를 확인한다.

```bash
cd ~/YOBA && nc -z -G 2 localhost 6379 && echo "터널 OK" || echo "터널 꺼짐 → 1번 다시"; uv run --env-file .env python -c "
import os,redis
r=redis.Redis.from_url(os.environ['YOBA_REDIS_URL'],socket_timeout=5)
try: r.ping(); print('Redis OK, YOBA 마지막 전송:', (r.get('yoba:live:webcam') or b'none')[:80])
except redis.AuthenticationError: print('Redis 비밀번호 틀림 → .env 확인')
except redis.RedisError: print('Redis 연결 실패 → 터널 확인')"
```

| 증상 | 할 일 |
|---|---|
| `터널 꺼짐` | 1번을 다시 한다. YOBA는 끄지 않아도 터널이 붙으면 다시 전송한다 |
| YOBA 로그에 `실시간 Redis 전송 실패` | 터널을 확인한다. 터널이 켜져 있으면 `.env`의 비밀번호를 확인한다 |
| CCTV 연결 실패 | 유선 케이블, CCTV 계정과 비밀번호를 확인한다 |
| 사람 수는 나오는데 화면이 빔 | 정지 장면이라 JPEG가 매번 같으면 웹이 화면을 오래된 것으로 판단한다. 카메라 앞에서 움직이면 나와야 한다 (웹 쪽 수정 필요, 미확인) |
| 웹 화면이 뚝뚝 끊김 | `configs/service.yaml`의 `redis.live_senders`를 늘린다 (동시 전송 수) |

## 4. (대안) 기존 10초 주기 모드

```bash
cd ~/YOBA && uv run --env-file .env yoba
```

- `configs/cameras.yaml`(카메라 주소와 구역)이 필요하다. 구역은 `scripts/roi_editor.py`로 그린다.
- 10초 모드는 `yoba:zone:*`, `yoba:img:*` 키에 쓴다 (spec 9.1). 웹이 이 키를 읽도록 바뀌기 전까지는 웹에 나오지 않는다.

## 5. 종료

1. YOBA 화면 창에서 `q`
2. 터널 창에서 `exit` (autossh면 `Ctrl+C`)
