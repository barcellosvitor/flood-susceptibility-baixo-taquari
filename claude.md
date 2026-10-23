Projeto: Suscetibilidade a alagamento com ML — cheia de maio/2024 (RS)

Projeto da disciplina de Pesquisa Aplicada (briefing Kunumi). Objetivo: reproduzir o núcleo metodológico de Alcântara et al. (2025) e aplicá-lo à cheia de maio de 2024 no RS, gerando um mapa de suscetibilidade, tabela de desempenho e ranking de importância das variáveis. Entrega final: artigo científico (dez/2026).

Idioma do projeto: português (código pode ter nomes em inglês).

Artigo de referência

Alcântara, E., Baião, C., Guimarães, Y., Mantovani, J. (2025). Machine learning and climate scenario integration reveals controls on flood susceptibility in the Taquari-Antas Basin, Brazil. Modeling Earth Systems and Environment, 11, 410. DOI: 10.1007/s40808-025-02585-8

5 modelos: Random Forest, Gradient Boosting, LightGBM, XGBoost, CatBoost.
23 variáveis ambientais; rótulo = mancha da cheia de setembro/2023 (Sentinel-1).
Melhor modelo: RF, acurácia 0,997 e ROC-AUC 0,999 (quase certamente inflado, ver abaixo).
Variáveis mais importantes: rx5day (14,25%), elevação (11,59%), rx1day (10,39%).
Depois aplicam o RF a cenários SSP com MIROC6 (fora do escopo deste projeto).
O artigo NÃO publica dados. Contato: enner.alcantara@unesp.br (autor correspondente); a análise de ML foi feita por Cheila Baião e o processamento espacial por Yasmim Guimarães.
Código de referência

Repositório: https://github.com/ennerha/Taquari-Antas_Flooding (arquivo Machine Learning, um notebook Colab exportado sem extensão). Cobre SÓ o treino e a comparação:

Lê X_block_*.npy / y_block_*.npy prontos do Google Drive (não publicados).
Remove NaN, aplica LabelEncoder e SimpleImputer, split 70/30 estratificado aleatório.
Treina os 5 modelos com 50 árvores/iterações e hiperparâmetros padrão.
Métricas: Accuracy, F1, Precision, Recall, ROC-AUC, MCC → model_comparison.csv.
Salva o modelo de maior AUC em best_model.pkl.

Lista de rasters declarada (24, o paper diz 23; provavelmente HillSha ficou de fora): EucDist_Drenagem, DEM, TPI, Landforms, LSFactor, Geologia, LULC_2023, CN_grid, Slope_Degree, Aspect_Planar, NDWI, NDVI, S_grid, ProfileCurvature, PlanCurvature, TopographicWetnessIndex, Solos, Geomorfologia, HillSha, e chuva do evento: rx1day_Sep23, rx5day_Sep23, cwd_Sep23, r10mm_Sep23, r20mm_Sep23 (índices ETCCDI).

O que falta no repositório: pré-processamento (alinhamento e empilhamento dos rasters, geração dos blocos), definição do rótulo, ordem das colunas de X, feature importance, predição do mapa e cenários SSP.

Problemas a corrigir na nossa versão
Split aleatório por pixel → vazamento por autocorrelação espacial (explica AUC 0,999). Usar validação espacial (GroupKFold / LeaveOneGroupOut por município ou bloco).
Modelo escolhido pelo próprio conjunto de teste → usar validação separada ou CV.
Variáveis categóricas (Geologia, Solos, Geomorfologia, LULC, Landforms) tratadas como números → one-hot, ou cat_features no CatBoost.
SimpleImputer é inócuo (NaN já removidos antes); imbalanced-learn instalado e não usado; use_label_encoder foi removido do XGBoost recente; imports não usados.
Checar balanço de classes: acurácia engana com classes desbalanceadas.
Decisões tomadas

Região (a confirmar com o orientador): Baixo Taquari — Muçum/Encantado até a foz no Jacuí (Roca Sales, Arroio do Meio, Lajeado, Estrela, Cruzeiro do Sul etc.). Faz parte da Região Hidrográfica do Guaíba, então segue no escopo do briefing.

Por quê:

Cheia fluvial em vale, com pouca interferência de diques/bombas (ao contrário de Porto Alegre e Canoas, onde falhas do sistema de proteção explicam parte do alagamento).
Escala de milhares de km²: índices de chuva variam no espaço e fazem sentido como variável.
Teve grandes cheias em set/2023 (evento do paper) e mai/2024 → permite validação entre eventos.
Mesma bacia e mesmas variáveis do paper → comparação direta.

Consequência para o artigo: a lacuna passa de "bacia nova" para "mesma bacia, novo evento, com validação entre eventos".

Pergunta de pesquisa: tratar o produto como SUSCETIBILIDADE (onde a água tende a chegar numa cheia de grande porte), não como previsão de eventos futuros. O experimento C é a forma honesta de testar "prever áreas futuras".

Dados

Rótulos:

mai/2024: mancha do MapBiomas RS Climate Emergency Toolkit.
set/2023: não há mancha pronta. Opções, em ordem: (1) checar dados de Mantovani et al. (2024), npj Natural Hazards 1:9, DOI 10.1038/s44304-024-00009-8 (open access); (2) pedir aos autores; (3) gerar no GEE com Sentinel-1 (antes vs. pico, limiar de retroespalhamento; ver tutorial share.gishub.org/brazil_floods).
Em ambos: subtrair água permanente (rios, lagos), senão o modelo aprende "onde tem água".

Variáveis (grade comum de 30 m):

Terreno: ANADEM (modelo de terreno, preferir ao Copernicus) → declividade, aspecto, TWI, TPI, curvaturas, LS factor, landforms.
HAND e distância à drenagem (derivar do ANADEM com WhiteboxTools ou pysheds).
Superfície: MapBiomas LULC 2023; NDVI/NDWI do Sentinel-2 ANTERIOR ao evento.
Solos, geologia, geomorfologia: IBGE.
CN / S (SCS Curve Number) a partir de grupo hidrológico do solo + LULC.
Chuva: grades contínuas (CHIRPS, MERGE/CPTEC ou ERA5-Land) em vez de interpolar estações do INMET. Calcular os índices ETCCDI POR EVENTO (set/2023 para treino, mai/2024 para teste).

Ferramenta principal: Google Earth Engine (Python: earthengine-api / geemap). Empilhar variáveis numa imagem, stratifiedSample balanceado dentro/fora da mancha, exportar CSV com coordenadas e código do município de cada ponto. Impor distância mínima entre pontos.

Experimentos
A. Split aleatório dentro de set/2023 — só para reproduzir os números do paper.
B. Leave-one-municipality-out em cada evento — validação espacial.
C. Treino em set/2023, teste em mai/2024 — resultado principal.
D. Repetir C sem as variáveis de chuva — o terreno sozinho generaliza melhor?

Métricas: Accuracy, F1, Precision, Recall, ROC-AUC, MCC (as do paper). No experimento C, priorizar ROC-AUC: mai/2024 foi maior que set/2023, então um limiar fixo de 0,5 tende a subestimar a extensão; o que importa é se o modelo ordena bem os pixels.

Saídas: mapa de suscetibilidade, mapa de diferenças (acertos / falsos positivos / falsos negativos) contra a mancha, ranking de importância das variáveis.

Cuidados
VAZAMENTO: nenhuma variável derivada de imagens durante o evento (NDWI/NDVI de maio/2024 enxergam a própria água do rótulo).
Sentinel-1 em vale encaixado tem sombra e layover nas encostas → inspecionar manchas e mascarar declividade alta ao amostrar.
Registrar a ordem das colunas e os nomes das variáveis em todo o pipeline.
A diferença entre as métricas do experimento A e as de B/C é uma contribuição do artigo, não um fracasso da reprodução.
Status e próximos passos

Cronograma do briefing: semana 5 (25/09) = obter mancha de referência e pré-processar variáveis; semana 6 (02/10) = datasets de treino/teste; semana 7 (09/10) = treino e avaliação; semana 8 (16/10) = importância; semana 9 (23/10) = mapa e validação espacial.

Scaffolding criado em 23/09/2026 (repo git inicializado, sem commits ainda — trabalhando só local, por pedido do usuário):

Estrutura: src/{01_aoi,02_labels,03_variables,04_sampling}.py (01 implementado e testado; 02-04 ainda stubs com TODO no docstring) + src/config.py (caminhos, municípios do Baixo Taquari, EE_PROJECT, eventos, constantes de amostragem); notebooks/; data/{raw,interim,processed}/; outputs/{figures,models}/.
requirements.txt (earthengine-api, geemap, geopandas, rasterio, whitebox, pandas/numpy/sklearn, imbalanced-learn, xgboost/lightgbm/catboost, shap, jupyter) + .venv local, tudo instalado e importa sem erro. .gitignore ignora dados, saídas e credenciais GEE.
README.md com instruções de setup e o mapa da estrutura — aponta para este arquivo para o contexto completo.

Earth Engine autenticado (conta pessoal — a institucional/gcloud bloqueou com "app blocked", resolvido usando `earthengine authenticate --auth_mode=notebook`, que evita abrir navegador embutido). Projeto do Google Cloud registrado no Earth Engine: `kunumi-alagamento-rs` (org `vitor-barcellos-org`), setado em config.EE_PROJECT e passado explicitamente em `ee.Initialize(project=...)` — o projeto default do gcloud não tinha a API do Earth Engine habilitada.

01_aoi.py concluído e validado. Versão inicial usava FAO/GAUL/2015/level2 (~1192 km²); trocado em 24/09/2026 pra `projects/mapbiomas-workspace/AUXILIAR/MUNICIPIOS/municipios-RS` (malha oficial que o próprio MapBiomas usa — achada nos scripts que o usuário copiou, ver abaixo). Nomes batem exatos (NM_MUN, sem normalizar acento) e a área fica mais precisa: **1049 km²** pros mesmos 7 municípios. AOI exportada em data/interim/aoi_baixo_taquari.geojson.

Bug de ambiente corrigido: console do Windows (cp1252) corrompia acentos no print (mojibake). config.py agora força stdout/stderr para utf-8 na importação — todo script que importa config herda o fix.

02_labels.py implementado (24/09/2026) via Sentinel-1 (change detection, método UN-SPIDER), rodando sem erro pros dois eventos — mas com uma limitação séria de dados, ver "ACHADO IMPORTANTE" abaixo. Dois bugs de GEE corrigidos no caminho, documentados porque vão se repetir em 03/04:

1. `.mosaic()` de uma ImageCollection perde a projeção real das imagens de origem (cai pra um identity transform degenerado, "1 grau por pixel"). Toda operação por pixel (focal_mean, ee.Terrain.slope, etc.) precisa de `.reproject(crs=CRS_TRABALHO, scale=...)` logo após o mosaic, senão roda sobre pixels de ~111 km e mascara tudo silenciosamente (sem erro). config.CRS_TRABALHO = EPSG:32722 (UTM 22S, mesma que o GEE atribui nativamente às cenas Sentinel-1 desta região).
2. `geemap.geojson_to_ee` não existe nessa versão (0.38.5) — usar `geemap.vector_to_ee`.

ACHADO IMPORTANTE — não existe NENHUMA imagem Sentinel-1 (COPERNICUS/S1_GRD, IW) no catálogo do GEE para esta região antes de 14/09/2023 (checado numa bbox bem maior que a AOI — gap regional, não do recorte). Provável efeito da falha do Sentinel-1B (dez/2021): só o 1A operando, ESA reduziu aquisição em áreas não-prioritárias como o interior do RS; a cobertura decola só depois do desastre. Isso vale pros dois eventos: nem set/2023 nem mai/2024 têm imagem pré-evento.

Workaround usado: comparar a passagem mais próxima do pico ("umido") contra uma referência seca de OUTRA época do mesmo evento ("seco", não necessariamente anterior no calendário) — ver config.JANELAS_SENTINEL1. Datas escolhidas também para casar orbitProperties_pass (ascending/descending têm geometria de visada diferente, some se não casar).

Resultado com esse workaround: mancha de set/2023 ~0,5 km² e mai/2024 ~0,1 km² — ORDENS DE GRANDEZA menor do que uma enchente catastrófica de verdade (essas cheias cobriram dezenas/centenas de km² e destruíram cidades inteiras). Decompondo o funil (set/2023): 7118 pixels acima do limiar → 837 depois de remover água permanente (JRC, perde 88%) → 793 depois da declividade → 559 depois da limpeza de blobs. Hipótese mais provável: a passagem mais próxima do pico disponível é 9-11 dias DEPOIS do pico, e cheia de rio de vale encaixado drena rápido — a maior parte da área alagada no pico já tinha baixado quando o satélite passou, sobrando só a faixa ribeirinha que o JRC já classifica como "água permanente" (por isso a máscara de água permanente come quase tudo). Não é bug, é limitação real dos dados disponíveis — não ajustei limiares pra forçar uma área maior (seria maquiar o resultado).

Consequência: a mancha via Sentinel-1, do jeito que os dados estão disponíveis pra essa região, provavelmente SUBESTIMA muito a extensão real — não deveria virar o rótulo principal sem essa ressalva bem explícita no artigo. Fortalece a prioridade de achar a mancha oficial (MapBiomas Toolkit ou Mantovani et al.) em vez de só confiar no Sentinel-1 autogerado.

RESOLVIDO em boa parte pra mai/2024 (24/09/2026) — usuário pediu pra tentar achar o asset por trás do MapBiomas RS Climate Emergency Toolkit mesmo sendo só um app (SPA) sem asset documentado.

Método 1 (eu sozinho): o HTML da página tem `init("https://mapbiomas-workspace.earthengine.app/javascript/rio-grande-do-sul-climate-emergency-toolkit-modules.json")` — esse JSON é o script compilado do app e expõe os assets reais em texto puro, sem precisar de navegador/login. Achei os 8 assets de "área atingida" por aí. Útil de lembrar: pra qualquer GEE App (`*.earthengine.app`), dá pra achar o script assim, sem claude-in-chrome.

Método 2 (usuário tem acesso ao Code Editor e copiou os scripts-fonte pra `mapbiomas/{2022,2024}/...` no repo — não consigo abrir o Code Editor sozinho, links tipo code.earthengine.google.com/<hash> exigem login). Lendo `mapbiomas/2024/toAsset-AREAS-AFFECTED` (o script que GERA os assets) e `mapbiomas/2024/toolkit-Great flood in RS` (o app), a metodologia real de "affected_area" é:

1. Índice de água via Sentinel-2 (SMA/frações gv-npv-soil-cloud, antes 01-25/04/2024 vs depois 30/04-18/05/2024, limiar de mudança ≤ -0.5).
2. Radar Sentinel-1, método UN-SPIDER (igual ao nosso!) — mas com baseline "antes" de 8 MESES (2023-09-01 a 2024-04-30, mosaico único), não uma janela estreita — evita o problema de esparsidade que a gente teve. VH, smoothing 50m, limiar 1.20 (descending) + limiar absoluto -18/-25dB (ascending), une os dois passes.
3. PCA (k-means, 8 clusters) sobre Sentinel-2, classes escolhidas via inspeção visual.
   Os 3 métodos combinados (OR) + limpeza de blobs (connectedPixelCount>=8) + máscara de água permanente (MapBiomas classes 26/31/33 + MODIS oceano, NÃO o JRC que a gente usa) + máscara de declividade (SRTM, <5°, não % — diferente do que eu implementei) + fusão com camada da Defesa Civil/IPH-UFRGS (`projects/mapbiomas-workspace/GT_EMCL/IPH_UFRGS_inundao_em_6_de_maio_de_2024`) + máscara HAND por bacia hidrográfica (dicionário `hand_per_basin` no script, com limiar específico por bacia — **'Taquari-Antas': 25** metros, direto relevante pra gente) + blend final com camada de deslizamento (perda de NDVI Planet NICFI + declividade>7°).

Achados adicionais nesse mesmo script/pasta, testados e funcionando:
- `projects/mapbiomas-workspace/AUXILIAR/MUNICIPIOS/municipios-RS` — malha municipal oficial (499 municípios, campos NM_MUN/CD_MUN/AREA_KM2/SIGLA_UF). Substituiu a FAO/GAUL no 01_aoi.py (ver acima).
- `projects/mapbiomas-workspace/AUXILIAR/BACIAS_RS/BH_RS_25mil` — bacias hidrográficas oficiais do RS (26 features, campo 'nome', inclui 'Taquari-Antas'). Não integrado ainda no pipeline, mas é a fonte certa se a gente quiser replicar a máscara HAND-por-bacia.
- `mapbiomas_climate_emergency-2024_RS-water-v1` — camada de água PURA (não "área atingida" geral), de uma versão mais antiga (v1) do processamento. Testada: **24,7 km²** na nossa AOI, bem menor que affected_area-v2 (149-178 km²) — provavelmente um produto intermediário/menos refinado (só um dos 3 métodos, sem a fusão completa). Não trocamos por ele ainda; ficou anotado como alternativa.
- Pasta `projects/mapbiomas-workspace/GT_EMCL/` (onde mora a camada da Defesa Civil) NÃO é listável pela API (permission denied em `ee.data.listAssets` — só dá pra acessar um asset se já souber o ID exato). Não consigo descobrir sozinho se existe uma camada de Defesa Civil equivalente pra set/2023 aí dentro. Se o usuário quiser checar, teria que navegar pelo Assets do Code Editor manualmente.

Usando `mapbiomas_climate_emergency-2024_RS-affected_area-v2` (a "Área atingida" geral, produto final da fusão dos 3 métodos + Defesa Civil + HAND): **170,1 km² atingidos** na AOI atualizada (1049 km²) — ordem de grandeza plausível pra essa catástrofe. 02_labels.py usa MapBiomas quando há asset em config.MAPBIOMAS_AFFECTED_AREA (hoje só mai/2024), Sentinel-1 como fallback (hoje só set/2023).

RESSALVA sobre essa mancha: é "área atingida" em geral — inclui o blend final com deslizamento, então não é 100% água. Mas a maior parte vem de 3 métodos de detecção de água/mudança, então é dominantemente um sinal de inundação. As sub-camadas por categoria (e o water-v1) estão em config.py se quiser algo mais específico. Mencionar essa definição no artigo.

Pendência que continua real: set/2023 ainda não tem fonte oficial equivalente (o toolkit é só do evento de 2024, e a pasta GT_EMCL não é listável pra achar uma Defesa Civil de set/2023 por conta própria). Enquanto isso não resolve, o rótulo de set/2023 é o Sentinel-1 com a ressalva de subestimação já documentada — não deveria virar resultado principal do artigo sem isso ficar bem explícito. Vale tentar replicar a metodologia dos 3 métodos (pelo menos o radar com baseline longo, que resolveria nosso problema de esparsidade) diretamente pra set/2023, já que agora temos o código de referência completo.

set/2023 melhorado em 24/09/2026 (a pedido do usuário, aproveitando o achado acima): trocamos a referência "seca" de uma única imagem pós-recessão (nov/2023) por uma MEDIANA de 14 cenas de ago-out/2021 (mesma estação, mesma órbita — checado antes de usar). Achado no caminho: há cobertura Sentinel-1 regular em 2021 (55 cenas, ~12 dias), mas ZERO entre jan/2022 e 13/09/2023 — bate exatamente com a falha do Sentinel-1B (dez/2021), e 2021 foi ano de seca no RS, então serve como baseline "normal" razoável. Resultado: área subiu de 0,5 para 1,1 km² — melhora pequena, ainda claramente subestimado. `_s1_vh_mosaic` virou `_s1_vh_composite` (usa `.median()` em vez de `.mosaic()`, mais robusto quando há várias cenas).

Bug de infraestrutura descoberto e corrigido: mediana de 14 cenas + focal_mean + connectedPixelCount é pesada demais pro download direto (`geemap.ee_export_image`/`getDownloadURL`) — erro real por trás do "An error occurred while downloading" do geemap é `"User memory limit exceeded"` (400, visível só chamando `getDownloadURL` manualmente e lendo o corpo do erro — geemap engole a mensagem). Isso vai se repetir em qualquer export pesado (03/04 inclusive). Criado `src/ee_utils.py` com `export_via_asset()`: materializa a imagem como asset temporário via `Export.image.toAsset` (export em lote, sem esse limite de memória), espera terminar (polling), baixa o asset já pronto (leve) e apaga o temporário. Usar isso — não `geemap.ee_export_image` direto — pra qualquer imagem com computação pesada (mediana de várias cenas, muitas bandas empilhadas, etc.).

Isso expôs mais uma pendência de permissão: `Export.image.toAsset` deu 403 "insufficient authentication scopes" com a autenticação atual (`--auth_mode=notebook`, que usamos pra fugir do bloqueio "app is blocked"). Falta rodar `earthengine authenticate --auth_mode=notebook --force --scopes=https://www.googleapis.com/auth/earthengine,https://www.googleapis.com/auth/cloud-platform` (pedido ao usuário, ainda não confirmado que rodou). Até isso rodar, qualquer script que precise de `export_via_asset()` (exports pesados) vai falhar com esse erro.

03_variables.py implementado (24/09/2026) — 16 variáveis, todas testadas e com valores fisicamente plausíveis nos dois eventos (ver mensagens da sessão para os números exatos): elevation, slope_degree, aspect, hillshade, tpi, profile_curvature, plan_curvature, dist_agua_permanente, lulc, ndvi, ndwi, rx1day, rx5day, r10mm, r20mm, cwd. `build_variable_stack(aoi, evento)` monta tudo — é isso que 04_sampling.py vai importar e usar; o módulo não exporta um raster completo (mesmo problema de memória do parágrafo acima), só faz uma checagem de sanidade quando rodado direto.

Bugs corrigidos nessa implementação (documentados no próprio código, mas resumindo aqui):
- Curvatura (Zevenbergen & Thorne 1987, via `neighborhoodToBands` numa vizinhança 3x3): tinha um sinal invertido no termo F (`.multiply(-1)` a mais) — corrigido.
- Distância à água: `fastDistanceTransform()` tem unidade ambígua (pixels vs. metros não é claro) — trocado por `.distance()` com `ee.Kernel.euclidean`. Tentei usar kernel em "meters" com raio de 10km → estourou o limite de 512px porque o kernel é dimensionado pela escala do REQUEST final (aqui sempre 30m), não da imagem. Tentei reprojetar a máscara pra uma grade mais grossa antes (pra caber um raio maior) → nearest-neighbor apagou quase toda a água esparsa, dando 98% dos pixels no teto não importa o raio. Voltei pra resolução nativa com raio de 7km (cabe no limite de 512px em 30m).
- **Achado real, não bug**: `seasonality` do JRC é mascarado (não zero) onde nunca houve água — só 2,1% dos pixels da AOI têm valor. Isso significa que "distância à água permanente" só enxerga o rio principal, não a rede de córregos pequenos — RENOMEADO de `dist_drenagem` pra `dist_agua_permanente` pra não fingir que é o EucDist_Drenagem do paper. O EucDist_Drenagem de verdade entra no mesmo grupo pendente do TWI/LSFactor (precisa acúmulo de fluxo).

Variáveis do paper de referência (24 total, ver "Código de referência" acima) — status atualizado:
- 16 confirmadas com código rodando (lista acima, incluindo o proxy dist_agua_permanente).
- 3 pendentes por precisarem de acúmulo de fluxo, sem fonte nova necessária, só mais implementação: TWI, LSFactor, Landforms — MAIS o EucDist_Drenagem de verdade (rede de drenagem, não só o rio principal). Decisão pendente: MERIT Hydro/HydroSHEDS (rápido, ~90m, GEE puro) vs. WhiteboxTools local sobre o DEM (mais preciso, 30m, mais trabalho — era o plano original).
- 5 bloqueadas por fonte externa não localizada: Solos, Geologia, Geomorfologia, CN_grid, S_grid (checamos convenção AUXILIAR/* do mapbiomas-workspace pra essas 3 — não existem lá; precisa vir do IBGE/CPRM de verdade).

Próximo passo imediato (status em 27/09/2026, antes do achado abaixo): (1) usuário rodar a autenticação com escopo de export (acima) e confirmar — CONFIRMADO AINDA NÃO FEITO, testei de novo em 27/09 e bate o mesmo erro 403; (2) decidir MERIT Hydro/HydroSHEDS vs. WhiteboxTools local pro grupo TWI/LSFactor/Landforms/EucDist_Drenagem; (3) 04_sampling.py (amostragem estratificada + export do CSV de treino).

Bloqueio real descoberto em 27/09/2026 (testado nesta data): a mancha de set/2023 via Sentinel-1 tem só 1.276 pixels alagados (30m) — sem nem aplicar a distância mínima entre pontos ainda. A config pedia 1.500 pontos "alagados" por evento (metade de 3.000 balanceados) — impossível com esse rótulo. Era a pendência nº1 antes do achado de 15/10 abaixo.

GRANDE ACHADO EM 15/10/2026 — contato direto com os autores do paper de referência (Alcântara et al.) deu certo: eles passaram os rasters originais usados no estudo, em `data/raw/taquari_antas_autores/` (pasta `Banco de Dados Paper Taquari-Antas` recebida, copiada pra lá). 22 GeoTIFFs cobrindo quase todas as 24 variáveis declaradas — e o mais importante: **o FloodMap real de 05/09/2023** (o rótulo de treino que era nosso maior problema).

Conteúdo da pasta (por subpasta):
- `FloodMap_SEP052023/`: FloodMap.tif (rótulo real, 0/1) + Landforms.tif
- `Sentinel 1 SAR/`: DEM, Slope_Degree, Aspect_Planar, HillSha, LSFactor, PlanCurvature, ProfileCurvature, TopographicWetnessIndex + outra cópia do FloodMap (idêntica byte a byte, confirmado)
- `Sentinel 2 MSI/`: NDVI, NDWI (específicos de set/2023)
- `Temáticos/`: Geologia, Geomorfologia, LULC_2023, Solos — todas com tabela de legenda (`.tif.vat.dbf`, lida com `dbfread`) com nome de cada classe
- `EucDist/`: EucDist_Drenagem (o de verdade, não o proxy de água permanente que a gente tinha), + bônus EucDist_Buildings, EucDist_Roads (não estavam na lista original de 24)
- `CN_S_grid/`: CN_grid, S_grid

Falta só TPI (não veio — fácil de calcular, já sabemos fazer).

Legendas extraídas das `.vat.dbf`:
- Geologia (12 classes): unidades geológicas nomeadas (Caxias, Depósitos aluviais, Formação Botucatu, Gramado, etc.)
- Geomorfologia (9 classes): inclui "Vales encaixados" (7.163.818 px), "Planícies fluviais ou flúvio-lacustres" (794.324 px) — direto relevante pra risco de alagamento
- Solos (16 classes): nomes técnicos completos (Brunizém Avermelhado, Cambissolo Bruno Húmico álico, etc.)
- LULC_2023 (13 classes): usa os MESMOS códigos do MapBiomas que já usamos em 03_variables.py (3=Forest Formation, 15=Pasture, 39=Soy Beans, etc.) — boa notícia, compatibilidade direta

PROBLEMA TÉCNICO investigado e resolvido: os arquivos declaram CRS EPSG:32718 (UTM zona 18S), o que é tecnicamente impossível pra essa região (zona 18S cobre Peru/Bolívia, não RS). Investigação: os números brutos de coordenada (easting ~2,65-3,05 milhões) são grandes demais pra UTM padrão (false easting sempre 500.000) em QUALQUER zona — intrigante. Testei transformar o ponto central sob 8 CRS candidatos (EPSG 32718, 5880, 5641, 5530, 29192, 32722, 31982, ESRI:102033) — só o declarado (32718) caiu perto do RS. Testei os 4 cantos: formam um quadrilátero visivelmente cisalhado (não devia, deveria ser ~retângulo numa projeção local de baixa distorção) — bate com "dado genuinamente calculado sob a fórmula da zona 18S" (muito longe do meridiano central -75°, por isso a distorção enorme e os números grandes), não um mislabel simples. Conclusão prática, validada: reprojetar com ferramenta padrão (rasterio.warp/GDAL) usando o CRS declarado (mesmo "estranho") recupera a geografia certa — testei reprojetando os bounds do FloodMap pra EPSG:32722 e pra EPSG:4326, e nossa AOI conhecida fica inteira dentro do resultado. CONCLUSÃO: pode usar os arquivos normalmente, só SEMPRE reprojetar com o CRS declarado (nunca assumir que os metros já estão certos sem reprojetar) — resampling nearest pra categóricas (Geologia/Solos/Geomorfologia/LULC/Landforms/FloodMap), bilinear pra contínuas (DEM, Slope, NDVI, etc.).

RESULTADO DECISIVO — reprojetei e recortei o FloodMap real pra nossa AOI (Baixo Taquari): **336,14 km² alagados** (373.484 pixels de 30m), contra ~1,1 km² do nosso Sentinel-1 autogerado — cerca de 300x mais. Resolve de vez o maior problema documentado do projeto (set/2023 subestimado). Pra contexto: 336 km² é ~32% da AOI inteira (1.049 km²) — plausível pra "a maior cheia em 150 anos" nessa região de vale.

Ressalva a documentar no artigo: 336 km² (FloodMap dos autores, set/2023) é MAIOR que os 170 km² do "affected_area" do MapBiomas pra mai/2024, apesar de mai/2024 ter sido descrito como o evento mais catastrófico — provavelmente reflete definições diferentes entre os dois datasets (FloodMap = inundação em si; affected_area = união de 3 métodos + Defesa Civil + HAND, mais conservador/específico). Comparação direta entre os dois eventos deve considerar isso, não é 1:1.

Ressalva adicional: nenhum dos 22 rasters está alinhado com os outros (extents diferentes por grupo de pasta — Geologia/Solos/Geomorfologia/CN/S/EucDist num extent, Sentinel 1 SAR/Landforms noutro maior, NDVI/NDWI noutro, LULC_2023 noutro, FloodMap no maior de todos). Confirma o que o claude.md já dizia faltar no repositório original do paper ("pré-processamento: alinhamento e empilhamento dos rasters") — vamos ter que fazer esse alinhamento nós mesmos (reprojetar+recortar+empilhar na nossa grade de 30m/EPSG:32722/AOI), que já era o plano do pipeline, só que agora alimentado com os rasters de verdade em vez de proxies.

DECISÃO TOMADA (15/10/2026, perguntado ao usuário): processar esses rasters LOCALMENTE com rasterio, não subir como asset do GEE. Motivo: evita a fricção acumulada com o GEE (limite de memória "User memory limit exceeded" recorrente, autenticação com escopo de export ainda quebrada, bugs de projeção do `.mosaic()`/`.distance()`/kernel) e o upload exigiria resolver a autenticação de export/storage primeiro mesmo. Arquitetura resultante:
- Variáveis ESTÁTICAS (terreno: DEM/Slope/Aspect/HillSha/LSFactor/Plan+ProfileCurvature/TWI/Landforms/EucDist_Drenagem/EucDist_Buildings/EucDist_Roads; temáticas: Geologia/Geomorfologia/Solos/CN_grid/S_grid) — mesmos rasters dos autores pros DOIS eventos (terreno/solo não muda entre set/2023 e mai/2024).
- NDVI/NDWI e LULC: autores (específicos de set/2023) pro evento de TREINO; continuam vindo do GEE (Sentinel-2/MapBiomas por evento, já implementado em 03_variables.py) pro evento de TESTE (mai/2024), porque são variáveis que mudam por evento e os autores só mandaram a versão de set/2023.
- Chuva (5 índices ETCCDI): continua vindo do CHIRPS via GEE pros dois eventos, sem mudança.
- Rótulo: TREINO passa a usar o FloodMap.tif real dos autores (em vez do Sentinel-1 fallback); TESTE continua usando o affected_area-v2 do MapBiomas (sem mudança).

Arquivos copiados pra `data/raw/taquari_antas_autores/` (pasta recebida preservada ali, estrutura original mantida).

`src/local_rasters.py` implementado e validado (15/10/2026): reprojeta/recorta cada raster pra nossa grade (AOI, EPSG:32722, 30m) com rasterio, nearest pra categóricas e bilinear pra contínuas, nodata vira NaN/-1 (nunca propaga os nodata originais variados — 127/255/2147483647 etc.). Grade alvo: 1318x1999 (2.634.682 pixels). Rodado e conferido:

FloodMap set/2023: 373.913 pixels alagados (~336,5 km², confirma o número de antes).
17 variáveis estáticas: 100% de cobertura (98-99% em Geologia/Geomorfologia/Solos, pequena borda sem dado — aceitável). Valores todos fisicamente plausíveis: elevação 9-686m, declive 0-58°, TWI 2,99-25,94, EucDist_Drenagem real 0-14.446m (bem mais estreito/plausível que o proxy JRC que tínhamos, que capturava só o rio principal). Validação bonita: S_grid bate EXATAMENTE com a fórmula SCS padrão S=25400/CN-254 (CN=100→S=0, CN=40→S=381) — confirma que entendemos certo os dois grids.
3 variáveis específicas de set/2023 (NDVI, NDWI, LULC): 100% cobertura, valores plausíveis.

Isso fecha a validação técnica — os dados estão prontos pra uso.

PIPELINE COMPLETO (15/10/2026, a pedido do usuário "faça os dois" — reescrever 02/03 E montar o 04):

`local_rasters.py` ganhou: TPI (calculado — DEM menos média focal numa janela quadrada ~9px via `scipy.ndimage.uniform_filter`, aproximação de janela quadrada do `focal_mean circle` que usávamos no GEE), `aoi_grid()` tornada pública, `reprojetar_arquivo()` generalizada pra aceitar qualquer Path (não só os rasters dos autores — reusada pra alinhar bandas baixadas do GEE também), `salvar_geotiff()`.

`02_labels.py` reescrito: set/2023 100% local (`local_rasters.build_flood_label_set2023()`, salva com `salvar_geotiff`); mai/2024 sem mudança (MapBiomas via GEE). Removido TODO o código do fallback Sentinel-1 (`_s1_vh_composite`, `_mascara_agua_permanente`, `_mascara_declividade`, `build_flood_mask`) — morto, substituído pelo FloodMap real. Testado: set/2023 local funciona perfeito (336,5 km²); mai/2024 falhou na autenticação (ver abaixo).

`03_variables.py` reescrito: cortado de 16 variáveis pra só o que não tem fonte local pros dois eventos — chuva (5 índices, ambos eventos) + NDVI/NDWI/LULC só de mai/2024 (`build_gee_stack(aoi, evento)`). Removidas as funções de terreno (`build_dem`, `terrain_stack`, `_curvaturas`) e `dist_agua_permanente` (proxy JRC) — tudo isso vem de `local_rasters.py` agora, com dado real em vez de proxy.

`04_sampling.py` criado: combina `local_rasters` (estáticas + TPI pros dois eventos; NDVI/NDWI/LULC/rótulo só set/2023) com GEE (`03_variables.build_gee_stack` pra chuva; `02_labels.build_flood_mask_mapbiomas` pro rótulo de mai/2024) — tudo baixado e realinhado na MESMA grade local antes de amostrar (função `_baixar_multibanda_gee`/`reprojetar_arquivo`, garante pixel a pixel idêntico não importa a origem). Amostragem: até `N_PONTOS_POR_EVENTO/2` por classe, no máx. 1 ponto por bloco de `DIST_MIN_ENTRE_PONTOS_M` (implementado como grade de blocos + embaralhamento + 1º candidato de cada bloco, não é nearest-neighbor exato mas é rápido e correto o suficiente), só pixels com TODAS as variáveis válidas.

Bug pego antes de rodar: minha máscara de validade inicial tratava QUALQUER valor -1 como nodata em qualquer banda — mas `aspect` usa -1 como valor legítimo (terreno plano, não "sem dado"). Corrigido: só bandas categóricas (`VARIAVEIS_CATEGORICAS`, construído a partir dos mesmos dicionários de `local_rasters.py`) usam o critério `!= -1`; contínuas usam `~isnan`.

TESTADO (15/10/2026):
- Caminho 100% local (`amostrar_evento(EVENTO_TREINO, usar_gee=False)`): **funcionou perfeito** — 3.000 pontos (1.500 alagados / 1.500 não-alagados, balanceamento perfeito batido com folga — 214.514 candidatos alagados disponíveis, bem mais que suficiente), 27 colunas, municípios variados e corretos (Estrela, Lajeado, Arroio do Meio, Cruzeiro do Sul, Encantado todos aparecendo). Prova que a lógica de amostragem/espaçamento/validade está certa.
- Caminho completo (`main()`, usar_gee=True nos dois eventos): FALHOU na autenticação do GEE — "Please authorize access to your Earth Engine account". Script tratou o erro corretamente (não quebrou, reportou os dois eventos como falha, não gerou CSV incompleto/enganoso).

BLOQUEIO DE AUTENTICAÇÃO (15/10/2026) — não é mais o erro de escopo insuficiente de antes, agora é "não autenticado" mesmo. Checado `~/.config/earthengine/credentials`: data de 23/09 (a autenticação ORIGINAL, `--auth_mode=notebook` sem escopo customizado), com escopos só leitura (`earthengine.readonly`, `cloud-platform.read-only`, `drive.readonly`, `devstorage.read_only`) — confirma que o pedido de reautenticar com escopo de export (feito em 24/09 e repetido em 27/09) nunca foi efetivamente concluído. E agora, ~3 semanas depois, nem a autenticação básica funciona mais (token expirado por inatividade). Ação pendente do usuário, igual antes, só que agora é ainda mais urgente (bloqueia TUDO que depende do GEE: chuva nos dois eventos, rótulo/NDVI/LULC de mai/2024):
```
earthengine authenticate --auth_mode=notebook --force --scopes=https://www.googleapis.com/auth/earthengine,https://www.googleapis.com/auth/cloud-platform
```

Próximo passo imediato: usuário rodar a autenticação acima; depois só rodar `python 04_sampling.py` de novo — o pipeline inteiro já está escrito e testado (a parte local 100%; a parte GEE só falta a credencial válida).

PIPELINE COMPLETO RODANDO DE PONTA A PONTA (16/10/2026) — usuário reautenticou (precisou de mais uma rodada: achou um erro extra "OAuth client is not fully created yet", que é propagação normal do Google após criar/alterar um client, resolvido só esperando e tentando de novo). Testei o escopo de export de novo: funcionou (`Export.image.toAsset` OK).

Rodei `04_sampling.py` com o GEE funcionando e achei mais um limite real: baixar várias bandas juntas num só `.tif` (via asset já materializado) estourou um limite DIFERENTE do de memória que já tínhamos resolvido — `"Total request size (89692000 bytes) must be less than or equal to 50331648 bytes"` (48 MiB). Esse é um teto de TAMANHO de resposta do `getPixels`/download direto, não de memória de computação — `export_via_asset` já resolve o de computação, mas o download final de um asset grande ainda pode estourar esse outro teto. Corrigido em `_baixar_multibanda_gee()`: baixa cada banda SEPARADAMENTE (pega `bandNames()` da imagem, um `export_via_asset` + reprojeção por banda) em vez de um `.tif` multi-banda só — cada banda sozinha fica bem abaixo do limite (~15-18 MB testado).

RESULTADO FINAL — os dois eventos rodaram 100%, do início ao fim:
- set/2023: 214.514 candidatos alagados / 883.983 não-alagados válidos -> amostrados 1.500/1.500 -> `data/processed/pontos_2023-09.csv`
- mai/2024: 161.715 candidatos alagados / 936.782 não-alagados válidos -> amostrados 1.500/1.500 -> `data/processed/pontos_2024-05.csv`

Validado: 3.000 pontos cada, 32 colunas (18 estáticas+TPI, NDVI/NDWI/LULC, 5 chuva, + evento/lon/lat/cod_município/município/flood), ZERO valores nulos, classes perfeitamente balanceadas, os 7 municípios todos representados nos dois eventos. Chuva plausível e consistente com a literatura: mai/2024 teve rx1day/rx5day bem maiores que set/2023 (71-103mm vs. 49-67mm; 203-256mm vs. 147-184mm) — bate com mai/2024 ser o evento descrito como mais catastrófico em termos de volume de chuva, mesmo nossa AOI específica tendo mostrado área alagada maior em set/2023 (métricas diferentes, não contraditório — ver ressalva já registrada acima sobre FloodMap vs. affected_area não serem 1:1).

Semana 6 do cronograma (02/10, "datasets de treino/teste") está CONCLUÍDA, com atraso mas de forma robusta. Próximo passo real agora: semana 7 (treino e avaliação dos modelos) — nenhum código de treino existe ainda. Falta escrever do zero: carregar os dois CSVs, implementar os experimentos A-D (A: split aleatório em set/2023; B: leave-one-municipality-out; C: treino set/2023 → teste mai/2024, resultado principal; D: C sem variáveis de chuva), treinar os 5 modelos (RF, Gradient Boosting, LightGBM, XGBoost, CatBoost) com tratamento correto de categóricas (Geologia/Solos/Geomorfologia/LULC/Landforms — one-hot ou cat_features nativo do CatBoost, ver "Problemas a corrigir" no topo deste arquivo) e as métricas definidas (Accuracy, F1, Precision, Recall, ROC-AUC, MCC).

SEMANA 7 CONCLUÍDA (20/10/2026) — `src/05_train.py` implementado e rodado de ponta a ponta nos 4 experimentos, usando os dois CSVs de `data/processed/`.

Decisões de implementação:
- `cn_grid`/`s_grid` são discretas por construção (lookup CN × uso do solo) mas numericamente ordenadas (maior CN = mais escoamento) — tratadas como variáveis **contínuas**, não entraram no one-hot.
- Categóricas nominais de verdade — `geologia`, `geomorfologia`, `solos`, `lulc`, `landforms` — recebem one-hot encoding (`OneHotEncoder(handle_unknown="ignore")`), igual pros 5 modelos (não só CatBoost), pra manter o ranking de importância comparável entre modelos. `handle_unknown="ignore"` é rede de segurança pro experimento C/D (treino e teste são eventos/fontes diferentes) e pro B (uma classe rara pode só existir no município retirado).
- 50 árvores/iterações (igual ao código de referência do paper, "hiperparâmetros padrão") nos 4 experimentos, pra manter A-D comparável entre si. Tuning fica pra depois, fora do escopo desta entrega.
- Saídas em `outputs/models/`: `model_comparison.csv` (tabela longa, 85 linhas — 5 modelos × (1 em A + 7 municípios×2 eventos em B + 1 em C + 1 em D)), `model_comparison_resumo.csv` (médias ± desvio, B agregado por município) e `best_model_experimento_c.pkl` (pipeline completo do melhor modelo do resultado principal, por ROC-AUC).

Resultados (primeira rodada, seed=42):
- **A (split aleatório em set/2023)**: AUC 0,995–0,998, Acc 0,97–0,98, MCC 0,94–0,96 em todos os 5 modelos. Alto, mas não tão extremo quanto o paper original (AUC 0,999/Acc 0,997) — ainda assim claramente inflado por vazamento espacial, confirma a hipótese do claude.md.
- **B (leave-one-municipality-out)**: cai, mas não tanto quanto esperado — set/2023 ainda AUC ~0,99/Acc ~0,96 (mesma correlação temporal dentro do evento, só removendo o município), mai/2024 cai mais pra AUC ~0,936–0,938/Acc ~0,87–0,88. Confirma que a maior parte do vazamento de A é espacial-de-vizinhança (resolvido por LOMO), mas ainda sobra alguma coisa específica do evento.
- **C (treino set/2023 → teste mai/2024, resultado principal)**: AUC 0,91–0,92, Acc 0,83–0,85, MCC 0,67–0,69 nos 5 modelos — bem mais honesto, queda real e esperada ao generalizar entre eventos diferentes. Melhor modelo por AUC: CatBoost (0,9220, empate técnico com XGBoost), salvo em `best_model_experimento_c.pkl`.
- **D (igual a C, sem as 5 variáveis de chuva)**: AUC cai um pouco (0,905–0,916, ~0,01 abaixo de C em cada modelo) — a chuva ajuda a ordenar os pixels. Mas achado interessante: o **MCC sobe** em 4 dos 5 modelos (ex. XGBoost 0,698 vs 0,669; LightGBM 0,694 vs 0,668; GradientBoosting 0,693 vs 0,683) — ou seja, no limiar fixo de 0,5 o terreno sozinho classifica tão bem ou melhor que com chuva, mesmo a chuva ajudando no ranking geral (AUC). Consistente com a nota do claude.md sobre priorizar ROC-AUC em C (mai/2024 foi maior que set/2023, limiar fixo tende a subestimar) — vale explorar no artigo como um resultado real, não um bug.

Pendência real a investigar antes de reportar como definitivo: Experimento A ainda ficou bem mais alto que B/C mesmo com split aleatório dentro do MESMO município (não só vizinhança entre municípios) — vale checar se `DIST_MIN_ENTRE_PONTOS_M=90` (3 pixels de 30m) é suficiente pra descorrelacionar espacialmente, ou se autocorrelação de curto alcance (ex. textura de relevo, chuva por pixel praticamente idêntica numa grade de 90m) ainda domina o split aleatório.

Próximo passo real agora: semana 8 (importância das variáveis) — nenhum código ainda. Extrair feature importance nativa dos modelos treinados em C (ou permutation importance, mais robusta e comparável entre modelos) e agregar as colunas do one-hot de volta pro nome da variável original (ex. somar/max das dummies de `geologia_1`, `geologia_2`, ... de volta pra "geologia") pra montar o ranking final.

SEMANA 8 CONCLUÍDA (20/10/2026) — `src/06_importance.py` implementado e rodado. Método: permutation importance (`sklearn.inspection.permutation_importance`, 10 repetições, scoring=ROC-AUC) sobre os 5 modelos do experimento C, medida no conjunto de TESTE (mai/2024) — mede o que cada variável contribui pra generalizar entre eventos, não só pro ajuste em set/2023. Como a permutação embaralha as colunas ORIGINAIS (antes do OneHotEncoder, que mora dentro do Pipeline), as categóricas são embaralhadas como uma coluna só, sem precisar agregar manualmente as dummies do one-hot — mais simples do que o plano inicial. Saídas: `outputs/models/feature_importance.csv` (por modelo) e `feature_importance_resumo.csv` (ranking consenso, média entre os 5 modelos).

RESULTADO E ACHADO IMPORTANTE: **elevação domina sozinha, 78-96% de importância em TODOS os 5 modelos** (RF 78%, GB 96%, LightGBM 93%, XGBoost 92%, CatBoost 93%). Nenhuma variável de chuva aparece no top 10 consenso — muito diferente do paper de referência (rx5day 14,25% em 1º, elevação só 11,59% em 2º). Top 10 consenso: elevation (90,6%), dist_drenagem (1,4%), tpi (1,3%), twi (1,2%), slope_degree (1,1%), solos (0,9%), geomorfologia (0,9%), ls_factor (0,7%), landforms (0,6%), geologia (0,5%).

Investigado o porquê (não é bug do código, é uma propriedade real dos dados): checada a variância espacial de cada variável nos 3.000 pontos amostrados.
- `r10mm` e `cwd` em set/2023 são **literalmente constantes** (1 valor único nos 3.000 pontos, std=0,000) — toda a AOI teve exatamente a mesma contagem de dias com +10mm e a mesma sequência de dias secos na janela de 30 dias. `r20mm` quase constante (8 valores distintos, std=0,459). Em mai/2024 varia um pouco mais (`r10mm` 11 valores, `cwd` 8 valores) mas ainda muito pouco. `rx1day`/`rx5day` têm mais textura (83-97 valores distintos) mas nada perto de `elevation` (2.996 valores distintos em 3.000 pontos, essencially contínua).
- Explicação mais provável: CHIRPS tem resolução nativa ~0,05° (~5,5 km); nossa AOI (Baixo Taquari, 1.049 km², ~30x35 km) cabe em poucas células CHIRPS, então os índices de chuva (calculados sobre uma janela regional) têm pouquíssima variação espacial DENTRO da AOI — ao contrário da bacia Taquari-Antas inteira do paper original (muito maior), onde a lógica registrada em "Decisões tomadas" ("índices de chuva variam no espaço e fazem sentido como variável") se aplicava com mais força. Permutation importance mede literalmente "o quanto embaralhar essa coluna piora o modelo" — pra uma coluna quase constante, embaralhar não muda quase nada, então a importância tende a zero por construção, não por a chuva ser irrelevante pra inundação.
- Isso reencaixa o achado do experimento D (semana 7): D (sem chuva) ficar quase igual a C não é só "o terreno generaliza bem" — é, em boa parte, "a chuva quase não varia espacialmente dentro da nossa AOI, então não há muito sinal pra perder". Ressalva importante pro artigo: nossa escala regional (sub-bacia pequena) limita o que os índices de chuva do paper original conseguem contribuir — explicar isso explicitamente evita que pareça um erro de implementação.

Próximo passo real agora: semana 9 (mapa de suscetibilidade + validação espacial) — nenhum código ainda. Falta: gerar o raster de probabilidade pra AOI inteira usando o `best_model_experimento_c.pkl`, o mapa de diferenças (acertos/falsos positivos/negativos) contra o FloodMap e o affected_area, e consolidar os resultados dos 4 experimentos numa tabela final pro artigo.

CORTE DE VARIÁVEIS (22/10/2026) — a pedido do usuário, depois do achado de importância da semana 8: testamos remover `r10mm`/`cwd` do conjunto de variáveis. Decisão e método documentados aqui porque mudam o pipeline principal pra frente.

Por quê cortar especificamente essas duas (e não outras com importância baixa): em set/2023 (evento de TREINO) `r10mm` e `cwd` são **exatamente constantes** — um único valor nos 3.000 pontos amostrados (ver achado da semana 8) — então nenhum modelo de árvore pode gerar um split com ganho positivo nelas, por construção matemática. Isso é verificável só com dado de treino, sem olhar mai/2024. `r20mm` ficou de fora do corte por ter variância real (pequena, mas não nula — 99,8% dos pontos em só 2 valores, mas não 100%).

Cuidado metodológico seguido: NÃO decidimos olhando o quanto cada variável "importava" no teste (mai/2024) — isso seria repetir, um nível acima, o mesmo erro já criticado no paper original (escolher modelo pelo próprio conjunto de teste). A decisão foi validada com leave-one-municipality-out DENTRO de set/2023 (nunca toca mai/2024): comparamos os 5 modelos com e sem as duas colunas, 7 folds. Resultado: LightGBM/XGBoost idênticos até a 4ª casa decimal, CatBoost idêntico, GradientBoosting praticamente igual; RandomForest variou na 3ª casa decimal (ruído do sorteio de subconjunto de variáveis por split, não sinal perdido). Confirma a previsão matemática — cortar colunas constantes não muda nada. Só depois, como checagem de transparência (não como critério de decisão), comparamos o experimento C completo vs. reduzido — mesmo padrão, diferença desprezível.

Implementado: `COLS_EXCLUIDAS = ["r10mm", "cwd"]` em `05_train.py`, removidas logo em `carregar_dados()` — `06_importance.py` herda automaticamente (importa `carregar_dados()` de `05_train.py`). Conjunto de variáveis passa de 26 pra 24. Re-rodei os dois scripts depois da mudança: `model_comparison.csv`/`model_comparison_resumo.csv`/`best_model_experimento_c.pkl` e `feature_importance.csv`/`feature_importance_resumo.csv` já refletem o conjunto limpo (resultados praticamente idênticos aos de antes do corte, como esperado — elevação ainda domina 81-96% nos 5 modelos). O script usado só pra essa comparação (`07_selecao_variaveis.py`) foi apagado depois de confirmar a decisão — o raciocínio e os números completos ficam registrados aqui, não precisa ficar um script que já cumpriu seu papel (e que daria erro se rodado de novo, já que as colunas não existem mais em `carregar_dados()`).

SEMANA 9 CONCLUÍDA (22/10/2026) — `src/08_mapa_suscetibilidade.py` implementado e rodado. Aplica o `best_model_experimento_c.pkl` (CatBoost, treinado em set/2023, 24 variáveis) em TODOS os pixels válidos da AOI (não só os 3.000 amostrados), pros dois eventos, e gera o mapa de diferenças contra o rótulo real de cada um. Reaproveita a mesma montagem de bandas de `amostrar_evento` (04_sampling.py) — por isso precisou rebaixar as mesmas bandas GEE (chuva pros dois eventos; NDVI/NDWI/LULC só mai/2024) banda por banda de novo, confirmando que a autenticação do GEE ainda estava válida (~1 semana depois do último uso).

Cobertura válida: **1.098.497 pixels (41,7% da AOI)** nos dois eventos — bate exatamente com a soma dos candidatos válidos já reportada em 04_sampling.py (set/2023: 214.514+883.983; mai/2024: 161.715+936.782, ambos somam 1.098.497), confirma que a máscara de validade do mapa é consistente com a da amostragem. A cobertura parcial (42% em vez de ~100%) já era esperada — ver "Ressalva adicional" de 15/10/2026 sobre os 22 rasters dos autores terem extents diferentes entre si (Geologia/Solos/Geomorfologia/CN/S/EucDist num extent menor que Sentinel 1 SAR/Landforms); a máscara final é a INTERSECÇÃO de todas as bandas, por isso fica bem mais restritiva que qualquer banda isolada.

Saídas em outputs/figures/: `suscetibilidade_<evento>.tif` (probabilidade contínua, float32), `diferencas_<evento>.tif` (0=acerto não-alagado, 1=acerto alagado, 2=falso positivo, 3=falso negativo, 255=inválido), `mapa_<evento>.png` (os dois lado a lado) e `resumo_mapas.csv`.

Resultado visual: o mapa de probabilidade é quase idêntico entre os dois eventos (esperado — elevação domina ~90%+ da importância e o terreno não muda entre eventos) — vermelho escuro concentrado no vale do rio, amarelo claro nas encostas/planaltos. Padrão fisicamente plausível, bate com a intuição de "suscetibilidade = proximidade/baixa elevação em relação ao rio".

Resultado numérico (limiar 0,5):
- **set/2023 (aplicado na própria AOI, evento de treino)**: acurácia por pixel 0,9778; TN=863.759, TP=210.322, FP=20.224, FN=4.192 → recall≈0,98, precisão≈0,91. Mapa de diferenças quase todo acerto (azul), pouquíssimo falso negativo — consistente com o LOMO do experimento B pra esse evento (AUC~0,99, Acc~0,96).
- **mai/2024 (resultado principal — generalização entre eventos)**: acurácia por pixel 0,8557; TN=805.901, TP=134.129, FP=130.881, FN=27.586 → recall≈0,83, mas precisão≈0,51 (quase metade dos pixels marcados como suscetíveis são falso positivo).

ACHADO IMPORTANTE — a acurácia de 0,8557 em mai/2024 é quase idêntica ao baseline trivial "sempre prever não-alagado" (936.782/1.098.497 = 0,8528): a classe é bem desbalanceada no mapa cheio (85% não-alagado vs. 15% alagado — diferente dos 3.000 pontos de treino/teste, que são balanceados 50/50 por construção). Confirma a ressalva que já estava no claude.md desde o início ("Checar balanço de classes: acurácia engana com classes desbalanceadas") — a métrica que importa aqui é ROC-AUC/recall, não acurácia bruta, e o mapa de diferenças deixa isso visualmente claro (muito vermelho/FP apesar da acurácia alta).

Padrão espacial do falso positivo em mai/2024: forma um "halo" visível ao redor do núcleo azul (acerto), seguindo o formato do vale — hipótese mais provável (ainda não verificada pixel a pixel): é a mesma área que alagou em set/2023 (336 km², evento maior) mas não chegou a alagar em mai/2024 (170 km² segundo o affected_area-v2) — ou seja, o modelo pode estar corretamente identificando terreno suscetível que esse evento específico não atingiu, não necessariamente errando. Bate com o enquadramento do projeto ("tratar o produto como SUSCETIBILIDADE, não como previsão de eventos futuros"). Verificação pendente: cruzar pixel a pixel os falsos positivos de mai/2024 contra o FloodMap de set/2023 pra confirmar quantitativamente a sobreposição.

VERIFICAÇÃO DO "HALO" (22/10/2026) — `src/09_verificacao_halo.py` implementado e rodado, confirmando quantitativamente a hipótese levantada na entrega da semana 9. Método: cruza pixel a pixel o mapa de diferenças de mai/2024 (`diferencas_2024-05.tif`) contra o FloodMap real de set/2023 (`local_rasters.build_flood_label_set2023()`) — os dois já estão na mesma grade (mesma `aoi_grid()`), sem reprojeção extra.

Resultado — hipótese confirmada com folga:
- **66% dos falsos positivos de mai/2024 caem dentro da mancha de set/2023** (86.307 de 130.881 px), contra só 19,5% da AOI válida estar dentro dessa mancha — enriquecimento de **3,4x**.
- Teste mais rigoroso (taxa de FP por zona, restrito a onde mai/2024 = não-alagado): na zona "alagou em set/2023 mas não em mai/2024" (79,2 km²), o modelo marca **98,1%** dos pixels como alagado (é basicamente tudo falso positivo ali); na zona "nunca alagou em nenhum dos dois eventos" (763,9 km², a maior parte da AOI), a taxa de FP cai pra **5,3%** — razão de **18,7x**.
- Visual (`outputs/figures/verificacao_halo_2024-05.png`): o "halo" ao redor do núcleo azul (acerto) é majoritariamente laranja (FP dentro da mancha 2023), concentrado perto do rio; o vermelho (FP fora de qualquer mancha, "erro genuíno") é bem mais esparso e espalhado.

Conclusão: a maior parte do que parecia "erro" do modelo em mai/2024 é, na verdade, o modelo prevendo corretamente uma área que É suscetível (alagou num evento maior, set/2023) e que esse evento específico, menor, simplesmente não atingiu. Isso é evidência concreta a favor do enquadramento do projeto ("SUSCETIBILIDADE, não previsão de evento") e deve virar uma figura/resultado explícito no artigo, não só uma nota de rodapé — métricas de "acerto" calculadas só contra o rótulo de um único evento subestimam a qualidade real do modelo.

TABELA FINAL CONSOLIDADA (22/10/2026) — `src/10_tabela_final.py` implementado e rodado. Junta `model_comparison_resumo.csv` (A-D), `feature_importance_resumo.csv`, `resumo_mapas.csv` e a verificação do halo numa entrega única pro artigo:
- `outputs/models/tabela_final_resultados_longa.csv` / `_wide.csv`: experimento (A, B-set2023, B-mai2024, C, D) x modelo x métricas (wide só tem AUC/Acc/MCC, as 3 priorizadas no projeto; longa tem as 6).
- `outputs/figures/comparacao_experimentos.png`: 2 painéis (ROC-AUC, MCC) com os 5 modelos lado a lado por experimento — mostra visualmente a queda A→B→C/D que é a própria contribuição do projeto (ver "Cuidados" no topo deste arquivo).
- `outputs/figures/importancia_variaveis_top10.png`: ranking consenso, 2 painéis (escala completa, dominada por elevation 91%; zoom nos ranks 2-10 pra ver a estrutura do resto).
- `outputs/figures/resumo_final_mapas.csv`: `resumo_mapas.csv` + as 3 métricas da verificação do halo (22/10/2026, ver acima) na mesma tabela.

Paleta e forma dos gráficos seguem a skill de dataviz do projeto (cores categóricas fixas por modelo, sem eixo duplo, grade discreta) — não é só estética, evita o modelo "ganhar" uma cor diferente em cada figura do artigo.

Pendências: confirmar a região com o orientador; resolver a mancha de set/2023 — opções em ordem: (a) replicar a metodologia dos 3 métodos do script MapBiomas com baseline longo pro radar, usando os assets auxiliares já testados (municipios-RS, BACIAS_RS, HAND); (b) checar a pasta GT_EMCL manualmente no Code Editor por uma camada de Defesa Civil de set/2023; (c) Mantovani et al. (DOI 10.1038/s44304-024-00009-8); (d) e-mail aos autores do paper de referência pedindo X_block/y_block; escrever o artigo (metodologia, resultados, discussão) usando os materiais já consolidados — essa é a próxima entrega real de código/análise em aberto.