# FutTrack: contexto para o Claude Code

Lê isto no início de cada sessão. No fim de cada sessão, atualiza a secção
**Estado atual** e o **Registo de sessões** e faz commit/push.

## O projeto
Visão computacional em Python: vídeo de futebol → deteção (YOLO) → tracking
(ByteTrack, `supervision`) → homografia (keypoints do campo) → posições em
metros → vídeo anotado com minimapa. Documentação detalhada no `README.md`
(em português de Portugal; manter assim).

- `main.py`: fluxo principal (todos os frames em memória).
- `yolo_inference.py`: fluxo por fases com exportação JSON (não usa o TeamAssigner).
- `trackers/tracker.py`: deteção, tracking e desenho.
- `team_assigner/team_assigner.py`: equipas, guarda-redes e árbitros pela cor.
- `reid/`: reidentificação (`global_id`) por equipa, tempo e posição em metros.
- `tests/`: testes sintéticos do ReID (`python -m pytest tests/`).
- `homografia/`: calibração (`homografia.py`), geometria do campo 120×70 m
  (`football_pitch.py`), minimapa (`minimap.py`).
- `modelo/player.pt` (deteção, classes player/goalkeeper/referee/ball),
  `modelo/pitch.pt` (32 keypoints, em **Git LFS**).

## Decisões tomadas
- **As classes do YOLO não são fiáveis** (dataset de treino pequeno). Por isso
  player/goalkeeper/referee são fundidos numa só classe de pessoa antes do
  tracking; a classe original fica em `yolo_class`, só como pista.
- **Equipas pela cor da camisola**, por track (não por frame): LAB com peso
  0,3 no L; relva estimada por frame (há equipas de verde-lima com o mesmo
  tom H da relva, só diferem no brilho); KMeans k=4 → os dois grupos maiores
  e distintos são as equipas; tracks longe de ambas são "outros".
- **NMS depois de fundir as classes** (IoU 0,7): sem isto, a mesma pessoa vinha
  em duas caixas (player + goalkeeper) e criava tracks duplicadas.
- **"Outros"**: guarda-redes = pessoa mais recuada do campo (atrás de ≥90 % dos
  jogadores em ≥60 % dos frames) e não junto à linha lateral; NÃO usar uma
  distância fixa à baliza (o GR sobe a x≈27 m no vídeo de teste). Equipa do GR
  = equipa mais próxima em x frame a frame. Restantes → árbitros.
- Numeração das equipas: pelo canal LAB em que as camisolas mais diferem
  (ordenar por L trocava as equipas entre execuções).
- **ReID** (`reid/`): tracklets → custo físico (mesma equipa/papel, sem
  sobreposição, ≤10 s, |início(B) − previsão(A)| ≤ 2 m + 8 m/s·dt) → algoritmo
  húngaro → rejeitar ligações ambíguas → `global_id`. GR sem limite de tempo.
  `lost_track_buffer=100` no ByteTrack.
- Cores no desenho: cor real da camisola da equipa; árbitros a amarelo;
  sem equipa a cinzento.

## Ambiente
- `supervision` tem de ser `<0.27`: as versões recentes removeram `sv.ByteTrack`.
- Nas sessões na cloud o `git lfs pull` falha, mas o `pitch.pt` pode ser
  descarregado com
  `curl -L -o modelo/pitch.pt https://media.githubusercontent.com/media/VascoCMH/Fut/main/modelo/pitch.pt`
  (verificar o sha256 com o ponteiro LFS). Depois, correr
  `git update-index --assume-unchanged modelo/pitch.pt`.
- Homografia em CPU: cerca de 254 s para os 750 frames, com todos calibrados.
  Para experiências, guardar em cache as deteções (`sv.Detections`) e as
  matrizes, e usar `Tracker.tracks_from_detections`.
- O vídeo de teste `input_videos/inputvideo.mp4` tem 1920×1080, 25 fps e 750
  frames. A deteção em CPU demora cerca de 28 s para 150 frames.

## Estado atual
- Feito: classificação por cor validada visualmente nos primeiros 150 frames
  do vídeo de teste (10 brancos, 10 verde-lima, árbitros e guarda-redes certos)
  e pelo Vasco no vídeo completo, com homografia, no computador dele (sem venv,
  com os pacotes instalados globalmente).
- Feito: ReID validado no vídeo completo com homografia: 25 identidades
  (10+10 jogadores, 1+1 GR, 3 árbitros) a partir de 30 tracks; 4 ligações
  corretas (verificadas visualmente), 0 ambíguas.
- Por fazer / ideias:
  - ReID com aparência (OSNet/torchreid) e número da camisola (OCR), para
    ausências longas e cortes de câmara. Detetar cortes de câmara.
  - Melhorar o modelo: mais dados (Roboflow Universe e frames próprios),
    `imgsz=1280`, ou retreinar só com 2 classes (`person`, `ball`).
  - Usar o TeamAssigner também no `yolo_inference.py` e no JSON.
  - Suavizar trocas de ID entre equipas (voto em janela temporal).
  - Se a cor falhar noutros jogos: embeddings (SigLIP/CLIP) + UMAP + KMeans.
  - Fixar versões das dependências (`requirements.txt`).

## Registo de sessões
- 2026-11-16: classificação de equipas, guarda-redes e árbitros
  (`team_assigner/`); classes de pessoa fundidas no tracker; minimapa com
  cores; criado este ficheiro. Validado localmente no vídeo completo;
  PR VascoCMH/Fut#1 para o main (integrado).
- 2026-11-16 (cont.): ReID (`reid/`, `global_id`), NMS após fusão de
  classes, nova regra de guarda-redes, numeração estável das equipas, testes.
