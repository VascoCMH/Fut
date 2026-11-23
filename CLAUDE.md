# FutTrack: contexto para o Claude Code

Lê isto no início de cada sessão. No fim de cada sessão, atualiza a secção
**Estado atual** e o **Registo de sessões** e faz commit/push.

## O projeto
Visão computacional em Python: vídeo de futebol → deteção (YOLO) → tracking
(ByteTrack, `supervision`) → homografia (keypoints do campo) → posições em
metros → vídeo anotado com minimapa. Documentação detalhada no `README.md`
(em português de Portugal; manter assim).

- `main.py`: fluxo principal em streaming (2 passagens: análise e desenho;
  nunca guarda o vídeo todo em memória).
- `yolo_inference.py`: fluxo por fases com exportação JSON (não usa o TeamAssigner).
- `trackers/tracker.py`: deteção, tracking e desenho.
- `team_assigner/team_assigner.py`: equipas, guarda-redes e árbitros pela cor.
- `reid/`: reidentificação (`global_id`) por equipa, tempo e posição em metros.
- `ball/`: limpeza e interpolação da bola, posse de bola e painel.
- `positions/`: estabilização da homografia e suavização das trajetórias.
- `tests/`: testes sintéticos do ReID e da bola (`python -m pytest tests/`).
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
  `lost_track_buffer=90` no ByteTrack (valor escolhido pelo Vasco).
- **Bola**: limpeza em píxeis ANTES da homografia (segmentos ≤3 frames
  incoerentes com ambos os vizinhos → falso positivo; recuperar outra
  candidata coerente); interpolação linear DEPOIS da homografia, em metros
  quando há matriz em toda a lacuna (≤50 frames), senão em píxeis.
- **Posse**: raio 2,5 m (1,5 m deixava 72 % dos frames "livres"); disputada se
  as duas equipas estão no raio com diferença <1 m; livre mantém a posse
  anterior; mudança só após 5 frames; interpolações de lacunas >8 frames NÃO
  decidem a posse (a bola falha sobretudo quando está tapada nos pés de
  alguém, e a linha reta passa a 3–4 m de quem a tem; visto no frame 664).
- **Streaming**: com 5 min (9000 frames) guardar os frames dava MemoryError
  (~56 GB). Passagem 1 em lotes (tracking, homografia, `TeamAssigner.add_frame`),
  depois equipas/ReID/bola/posse só com as tracks; passagem 2 desenha e escreve
  frame a frame (`Tracker.draw_frame`). A limpeza da bola corre depois da
  homografia, e as posições da bola são recalculadas com as matrizes guardadas.
  Painel de posse recebe contagens acumuladas (não a lista: seria O(n²)).
  Não voltar a guardar listas de frames no `main.py`.
- **Posições**: o ruído bruto era enorme (14 % das velocidades >11 m/s).
  (1) `HomographyStabilizer` logo após a análise: por frame, semelhança
  RANSAC das posições para a mediana temporal de cada pessoa → corrige a
  MATRIZ (não as posições) e reprojeta tudo. (2) `PositionSmoother` depois do
  ReID, por global_id: descarta caixas na borda (exceto se a track só tiver
  dessas) e saltos (Hampel ±7, max(3σ,1 m)); linha reta nos buracos e nas
  ausências ligadas; Savitzky–Golay 1 s grau 2 → position_smooth/speed_ms.
  speed_reliable=False nas pontas (derivada extrapolada). Posições estimadas
  em tracks["estimated"]: minimapa (círculo vazio) sim, estatísticas NÃO.
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
  frames. A deteção em CPU demora cerca de 28 s para 150 frames. Os vídeos
  (entrada e saída) não estão no repositório (`.gitignore`), por isso nas
  sessões na cloud é preciso fornecê-lo à parte. O vídeo de teste ainda está
  no histórico do git: `mkdir -p input_videos && git show
  17dcba3~1:input_videos/inputvideo.mp4 > input_videos/inputvideo.mp4`.

## Estado atual
- Feito: classificação por cor validada visualmente nos primeiros 150 frames
  do vídeo de teste (10 brancos, 10 verde-lima, árbitros e guarda-redes certos)
  e pelo Vasco no vídeo completo, com homografia, no computador dele (sem venv,
  com os pacotes instalados globalmente).
- Feito: ReID validado no vídeo completo com homografia: 25 identidades
  (10+10 jogadores, 1+1 GR, 3 árbitros) a partir de 30 tracks; 4 ligações
  corretas (verificadas visualmente), 0 ambíguas.
- Feito: bola e posse no vídeo completo: bola em 750/750 frames (592
  detetados, 5 falsos positivos removidos, 161 interpolados); posse 44 % /
  49 % / 6 % disputada; deteções verificadas visualmente (frames 648–720).
- Feito: `main.py` em streaming; mesmos resultados no vídeo de teste; pico
  de memória 3,5 GB (antes ~9 GB), cresce ~0,2 GB por 750 frames. Tempo em
  CPU: 563 s para 30 s de vídeo (5 min em CPU ≈ 1,5 h; com GPU muito menos).
- Feito: estabilização + suavização: velocidade fiável med 2,5 / p99 8,3 /
  máx 10,9 m/s; 142 frames corrigidos; GR fora de imagem estimado no minimapa.
  Posse continua a usar position_m (não a suavizada) — mesmos resultados.
- Por fazer / ideias:
  - Ausências longas: deslocar o jogador com o bloco da equipa (posição
    relativa ao centro), em vez da linha reta. Depende do ReID novo.
  - Estatísticas físicas (distância, sprints, mapas de calor) com
    position_smooth e só speed_reliable + measured.
  - Posse: validar com anotação manual de alguns segundos; usar a velocidade
    da bola para distinguir passe de condução; bola no ar (a projeção para
    metros só vale com a bola no chão).
  - ReID com perfil tático (posição relativa à forma da equipa) + conjunto
    fechado (≤11 por equipa) + aparência, combinados num só custo. Precisa
    de um vídeo mais longo para validar.
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
- 2026-11-17: bola (`ball/`): limpeza de falsos positivos, interpolação em
  metros, posse de bola com painel; o tracker guarda a confiança e as
  candidatas da bola. Ideia do Vasco para o ReID (perfil tático) registada.
- 2026-11-17 (cont.): `main.py` em streaming por causa do MemoryError com um
  vídeo de 5 min. Regras do Vasco para o ReID (equipa+classe, plantel
  fechado, único no raio, aparência + posição média) discutidas, por fazer.
- 2026-11-21: merge do `main` (commits "Fine Tuning" do Vasco: `.gitignore`
  novo em UTF-8, vídeos e resultados de treino fora do repositório,
  `lost_track_buffer=90`) no branch; conflitos resolvidos (`.gitignore` do
  Vasco; `Tracker` parametrizado com 90 por defeito). ReID atual NÃO serve
  para um jogo completo (limite de 10 s, cortes de câmara, escala das tracks).
- 2026-11-23: `positions/` (estabilização da homografia e suavização, com a
  ideia do Vasco de remover saltos e ligar ausências em linha reta).
