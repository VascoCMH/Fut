# Fut

## Homografia

A homografia converte uma posição na imagem, em píxeis, numa posição no
plano do campo, em metros. A implementação está em `homografia/homografia.py`.

Para executar a calibração de um vídeo, num ambiente Python com
`ultralytics`, `supervision`, `opencv-python` e `numpy` instalados:

```sh
python yolo_inference.py
```

O script usa `modelo/pitch.pt` para calibrar cada frame de
`input_videos/inputvideo.mp4` e `modelo/player.pt` para detetar jogadores e
guarda-redes. Guarda o vídeo em `output_videos/pitch_homography.mp4`, com um
minimapa no canto inferior direito. Podes escolher outro vídeo ou destino:

```sh
python yolo_inference.py --source input_videos/inputvideo1.mp4 --output output_videos/minimap.mp4
```

O processamento é dividido em fases completas:

1. Analisar todo o vídeo com o `Tracker` existente (YOLO + ByteTrack),
   guardando as bounding boxes e IDs por frame. A mesma instância do tracker
   mantém o estado entre lotes de 32 imagens.
2. Voltar a percorrer o vídeo, detetar o campo, calcular uma homografia por
   frame e projetar os pés dos jogadores identificados na primeira fase.
3. Guardar a lista completa em JSON, por defeito
   `output_videos/pitch_homography.json`. Cada frame contém os jogadores com
   `track_id`, `bbox`, `feet_px` e `position_m`, além da homografia e estado
   da calibração. As posições são em metros; ficam a `null` quando não é
   possível projetar. `frame_index` começa em zero.
4. Percorrer o vídeo novamente e desenhar apenas com os dados dessa lista,
   sem executar modelos de deteção ou tracking.

O terminal mostra a hora de conclusão em UTC e a duração de cada fase.
As imagens são lidas em lotes ou uma a uma; apenas os dados dos resultados
ficam em memória para todo o vídeo. Usa `--positions caminho.json` para
escolher o destino da lista. O JSON também regista FPS e dimensões do campo.

O minimapa projeta o centro inferior de cada bounding box (os pés) para o
campo. Os números amarelos são os IDs atribuídos pelo ByteTrack, iguais no
vídeo e no mapa. O tracker pode perder uma identidade após oclusões longas
ou cortes de câmara; não é reconhecimento da identidade real do jogador.
As cores não representam equipas. Compara as
posições com as linhas reais: um jogador junto ao meio-campo deve aparecer
junto à linha central no mapa. Saltos ou deslocamentos incoerentes podem
indicar erros na calibração ou na deteção dos pés. Isto permite uma inspeção
visual, mas não mede por si só a precisão em metros.

O contador `Fora` indica projeções fora dos limites do campo: esses pontos
não são desenhados nem encostados às linhas. Se a calibração falhar, o mapa
mostra `sem calibracao` e não reutiliza as posições do frame anterior.

### Como funciona

1. O YOLO deteta pontos de referência do campo na imagem: cantos,
   interseções de linhas, etc.
2. Cada ponto é associado à sua posição conhecida em `FootballPitch`.
   O índice **0** do YOLO corresponde ao ID **1** do campo, até ao índice
   **31**, que corresponde ao ID **32**. Confirma esta ordem no teu dataset;
   ter 32 pontos por si só não garante que as correspondências estejam certas.
3. `fit_yolo` ignora pontos com confiança inferior a 0.5, coordenadas
   inválidas ou posição `(0, 0)` usada como ponto ausente.
4. O OpenCV estima uma transformação com RANSAC, que tenta rejeitar
   correspondências inconsistentes. São precisos pelo menos quatro pontos
   numa configuração não degenerada; usa pontos bem espalhados pelo campo.
   O limiar predefinido é de 5 píxeis de erro de reprojeção na imagem.
5. A matriz resultante transforma outros pontos da imagem para o campo.

Para um ponto `(u, v)` da imagem, calcula-se:

```text
[a, b, c]ᵀ = H × [u, v, 1]ᵀ
x = a / c
y = b / c
```

`H` é uma matriz 3 × 3. A divisão por `c` permite corrigir a perspetiva:
distâncias iguais no campo podem ocupar quantidades diferentes de píxeis
consoante estejam mais perto ou mais longe da câmara.

### Usar com jogadores

```python
from ultralytics import YOLO
from homografia import Homography

result = YOLO("modelo/pitch.pt").predict("input_videos/TestePitch.jpg")[0]
homography = Homography().fit_yolo(result)

# Exemplo de uma bounding box de jogador nesta mesma imagem, em píxeis.
x1, y1, x2, y2 = 400, 200, 440, 320
feet = [(x1 + x2) / 2, y2]
x_m, y_m = homography.transform_points([feet])[0]
print(f"Posição no campo: ({x_m:.2f}, {y_m:.2f}) m")

# Transformação inversa: campo -> imagem.
pixel_position = homography.transform_points([[x_m, y_m]], inverse=True)[0]
```

Também podes calibrar manualmente com
`Homography().fit({1: (u1, v1), 6: (u6, v6), 25: (u25, v25), 30: (u30, v30)})`,
substituindo as variáveis pelas posições desses quatro cantos na imagem.

As coordenadas seguem o teu `FootballPitch`: origem no ponto 1, `x` de 0 a
120 m e `y` de 0 a 70 m. O ficheiro guarda os valores originais em centímetros;
`get_keypoints()` converte-os para metros. Confirma as dimensões e todas as
posições de referência para o campo/dataset que estás a usar.

Usa a posição dos pés porque a transformação só é válida para o plano do
relvado. O centro do corpo ou uma bola no ar não pertencem a esse plano.
Recalibra para cada imagem quando houver movimento ou zoom da câmara.
O script de vídeo recalibra em cada frame e projeta as posições dos jogadores
com tracking calculadas para esse mesmo frame.
