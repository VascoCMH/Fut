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
- **"Outros"**: junto a uma baliza (homografia) ou marcados como goalkeeper
  pelo YOLO → guarda-redes, com a equipa mais próxima em x frame a frame;
  restantes → árbitros (movidos para `tracks["referees"]`).
- Cores no desenho: cor real da camisola da equipa; árbitros a amarelo;
  sem equipa a cinzento.

## Ambiente
- `supervision` tem de ser `<0.27`: as versões recentes removeram `sv.ByteTrack`.
- Nas sessões na cloud o `git lfs pull` falha, por isso não há `pitch.pt` nem
  homografia. A lógica de cor pode ser testada sem homografia, porque os
  guarda-redes passam a ser decididos em píxeis.
- O vídeo de teste `input_videos/inputvideo.mp4` tem 1920×1080, 25 fps e 750
  frames. A deteção em CPU demora cerca de 28 s para 150 frames.

## Estado atual
- Feito: classificação por cor validada visualmente nos primeiros 150 frames
  do vídeo de teste (10 brancos, 10 verde-lima, árbitros e guarda-redes certos)
  e pelo Vasco no vídeo completo, com homografia, no computador dele (sem venv,
  com os pacotes instalados globalmente).
- Por fazer / ideias:
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
  PR VascoCMH/Fut#1 para o main.
