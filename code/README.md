# Código del artículo — Sistema de inferencia difusa basado en reglas de árboles para la predicción de la velocidad de corrosión

Implementación autónoma del método ganador y de su evaluación honesta.
Todo el análisis del artículo se reproduce ejecutando cuatro scripts en
orden. Requiere las librerías ya instaladas en el entorno del proyecto:
`numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`.

---

## 1. El método en pocas palabras

El objetivo es estimar la velocidad de corrosión interna $v$ [mpy] con
un modelo **continuo, auditable y coherente con los límites normativos**
(2 y 8 mpy). El método tiene cinco etapas:

1. **Segmentación normativa.** La variable objetivo se discretiza en
   tres clases: baja $(0,2]$, media $(2,8]$ y alta $(>8)$ mpy. Para la
   regresión, $v$ se satura en 20 mpy (2.5 veces el límite superior):
   por encima de ese nivel toda decisión de integridad es idéntica.
2. **Árboles binomiales por límite.** Se entrena un árbol de decisión
   por cada límite normativo: $T_2$ responde "¿$v > 2$?" y $T_8$
   responde "¿$v > 8$?". Los hiperparámetros (profundidad, tamaño
   mínimo de hoja, poda, umbral de probabilidad de la hoja) se
   seleccionan por validación cruzada interna maximizando el recall
   medio por clase más la mitad de la razón de recalls.
3. **Extracción de reglas.** Cada camino raíz→hoja es una regla
   SI/ENTONCES. Los caminos negativos de $T_2$ → consecuente *low*;
   los positivos de $T_2$ → *mid*; los positivos de $T_8$ → *high*.
   Se colapsan las hojas hermanas de la misma clase (particiones
   redundantes), se fusionan umbrales casi idénticos (1 % del rango
   robusto p1–p99) y se simplifican las condiciones de cada regla a un
   intervalo por variable. Las reglas *mid* llevan además la exclusión
   ordinal "Y NO (condiciones severas)".
4. **Funciones de pertenencia por intervalo.** Los umbrales de los
   árboles particionan el rango de cada variable. Cada intervalo
   recibe un término lingüístico cuya pertenencia es el mínimo de dos
   sigmoides ancladas en los umbrales: cruza 0.5 exactamente en el
   umbral (trazabilidad árbol→difuso) y su anchura de transición es la
   distancia del umbral a la mediana de los datos de entrenamiento
   dentro del intervalo, multiplicada por un factor de escala que se
   selecciona en los pliegues internos.
5. **Consecuentes e inferencia.** Cada regla tiene un consecuente
   singleton $c_r$ [mpy], inicializado en la mediana de su clase y
   ajustado por mínimos cuadrados con restricciones de caja: $c_r$
   no puede salir del rango normativo de su clase. La salida es el
   promedio ponderado $\hat v = \sum_r w_r c_r / \sum_r w_r$
   (Mamdani con salidas singleton, equivalente a Sugeno de orden
   cero), donde $w_r$ es el mínimo de las pertenencias de las
   condiciones de la regla.

**Protocolo de evaluación (anti-fuga).** Un StratifiedKFold externo de
5 pliegues (semilla 42, estratificado por la clase normativa) es
compartido por todos los métodos. Dentro de cada pliegue de
entrenamiento externo se repite TODA la cadena (selección de árboles,
selección del factor de escala, extracción de reglas, construcción de
pertenencias y ajuste de consecuentes). Las métricas se calculan solo
sobre las predicciones out-of-fold. Los benchmarks (regresión lineal,
árbol de regresión, SVR) usan exactamente los mismos pliegues.

---

## 2. Estructura de carpetas

```
code/
├── data/
│   ├── int_canolimon.csv    datos de campo crudos
│   └── processed.csv        datos limpios con nomenclatura física (script 1)
├── results/                 tablas intermedias en CSV (scripts 1-3)
├── src/                     los módulos y scripts descritos abajo
└── README.md                este archivo
```

Las figuras y tablas del artículo se escriben directamente en
`../figures/` y `../tables/`.

---

## 3. Módulos

### `nomenclature.py` — nomenclatura física
Única fuente de verdad de la equivalencia columna cruda → símbolo
físico. Ninguna figura, tabla o regla usa nombres de código.

| Objeto | Descripción |
|---|---|
| `NOMENCLATURE` | dict: columna cruda → (símbolo corto, símbolo mathtext, símbolo LaTeX, significado, unidad). Ejemplo: `velcorrcal_cmpy → v_mod` (tasa histórica de pérdida de espesor: $(e-e_{min})/t_s$, calculada en la construcción de la base de datos a partir del registro de inspección). |
| `RENAME` | dict para renombrar el dataframe al cargarlo. |
| `MEANING`, `UNIT` | significado y unidad por símbolo corto. |
| `mt(name)` / `lx(name)` | símbolo mathtext (figuras) / LaTeX (tablas) de una variable; devuelven el nombre si no está mapeado. |

### `fis_core.py` — el método propuesto
Autocontenido; solo usa numpy/pandas/scipy/sklearn.

| Función / clase | Qué hace (entradas → salidas) |
|---|---|
| `to_levels(v, edges)` | discretiza $v$ en clases 0/1/2 por los cortes normativos. |
| `recall_metrics(y, ŷ)` | → (recall medio, razón min/max, recalls por clase). |
| `collapse_tree(tree, leaf_thr)` | estructura simplificada del árbol: fusiona recursivamente hojas hermanas de la misma clase. `leaf_thr` umbraliza la fracción positiva ponderada de la hoja. |
| `extract_rules(tree, feats, positive, leaf_thr)` | caminos raíz→hoja del árbol colapsado cuya hoja vota por `positive` → lista de reglas `{conds, n}`. |
| `build_rulebase(t2, t8, feats, leaf_thr)` | reglas de ambos árboles con su nivel de severidad (0/1/2). |
| `merge_thresholds(rules, X, tol)` | fusiona umbrales de una variable más cercanos que `tol` × rango robusto (percentiles 1–99). |
| `simplify_rules(rules)` | combina las condiciones de cada regla en un intervalo por variable y elimina reglas con intervalo vacío. |
| `feature_partitions(rules)` | variable → umbrales únicos ordenados de la base de reglas. |
| `rules_readable(rules, symbols)` | reglas SI/ENTONCES legibles. |
| `sigmf(x, c, s)` | sigmoide logística centrada en `c`. |
| `mf_interval(x, lo, hi, data, span, scale)` | pertenencia del intervalo `(lo, hi]`: mínimo de dos sigmoides que cruzan 0.5 en los umbrales; anchura = distancia umbral→mediana de `data` × `scale`, acotada por el `span` robusto. |
| `FISModel` | sistema ajustado. Métodos: `predict(X)` (promedio ponderado de singletons), `activation_matrix(X)` (fuerzas de disparo n×R, con la exclusión ordinal de las reglas *mid*). |
| `fit_fis(Xtr, vtr, edges, tree_params, shoulder_scale, ...)` | ajusta la cadena completa SOLO con los datos recibidos (sin fuga dentro de un pliegue) → `FISModel`. Paso 4 interno: ajuste de consecuentes con `scipy.optimize.lsq_linear` (cotas por rango normativo, regularización hacia la mediana de clase). |
| `TREE_GRID` | rejilla de hiperparámetros de los árboles (54 combinaciones). |
| `select_tree_params(Xtr, vtr, edges, grid, n_inner)` | selección por límite en CV interna (recall medio + 0.5·razón). |
| `SCALE_GRID`, `select_shoulder_scale(...)` | selección del factor de anchura de transición en CV interna (por MAE). |
| `nested_cv_fis(X, v, ...)` | protocolo completo: 5 pliegues externos, toda la selección en pliegues internos → predicciones OOF, métricas por pliegue, parámetros elegidos por pliegue. |

### `style.py` — estilo gráfico
Toda figura deriva sus colores de la escala rojo/teal de la matriz de
correlación del proyecto: tonos teal para la clase baja, rojo claro
para la media, rojo oscuro para la alta, gris para lo neutro.

| Objeto | Descripción |
|---|---|
| `REDS`, `TEALS` | la escala fuente. |
| `LEVEL_EDGE/FILL/NAME` | color de borde/relleno por clase de severidad. |
| `METHOD_COLOR` | color de cada uno de los cuatro métodos comparados. |
| `corr_cmap()` / `heat_cmap()` | colormap discreto rojo/teal (matrices). |
| `save_fig(dir, name)` | guarda PDF + PNG, sin título dentro de la figura (los pies de figura viven en el artículo). |

### `diagrams.py` — diagramas esquemáticos
Cajas redondeadas ajustadas al texto, rombos de decisión y conectores
en L (estilo de los diagramas de arquitectura del proyecto).

| Función | Qué hace |
|---|---|
| `autobox(ax, cx, text, fc, ec, ...)` | caja redondeada ajustada al texto; ancla por centro o borde superior. |
| `autodiamond(...)` | rombo de decisión ajustado al texto. |
| `pt(nodo, lado)` / `arrow` / `connect` / `elbow` | puntos de borde, flechas rectas y conectores en L entre nodos. |
| `glossary(fig, pares)` | pie de figura "símbolo = significado". |
| `draw_tree(root, feats, mt, ...)` | diagrama horizontal del árbol COLAPSADO (`collapse_tree`), con ramas True/False etiquetadas y hojas coloreadas por severidad; coincide 1:1 con la base de reglas. |

---

## 4. Secuencia de ejecución

Ejecutar desde `code/` (cada script imprime su progreso):

```bash
python src/1_prepare_data.py
python src/2_run_nested_evaluation.py
python src/3_final_system.py
python src/4_make_figures_tables.py
```

**`1_prepare_data.py`** — carga `data/int_canolimon.csv`, elimina las
columnas derivadas del objetivo (desgaste, vida remanente, espesor
mínimo, años) y los 6 registros con velocidad negativa, satura $v$ en
20 mpy, renombra a la nomenclatura física → `data/processed.csv`,
`results/class_distribution.csv`, `results/feature_stats.csv`.

**`2_run_nested_evaluation.py`** — el protocolo anidado completo:
(1) árboles binomiales → `results/tree_metrics.csv`,
`results/confusion.csv`; (2) FIS propuesto →
`results/fis_params_by_fold.json`; (3) benchmarks en los mismos
pliegues; (4) resumen → `results/method_folds.csv` (métricas por
pliegue de los cuatro métodos), `results/oof_predictions.csv`,
`results/comparison.csv`.

**`3_final_system.py`** — reajusta la cadena con TODOS los datos (solo
para los artefactos interpretables; ningún número de desempeño sale de
aquí) → `results/rules.csv`, `results/deployed_params.json` y las
figuras `tree_th2/8`, `mf_three_panel`, `consequent_singletons`,
`rule_coverage`, `response_curve`.

**`4_make_figures_tables.py`** — figuras de evaluación
(`nested_cv_scheme`, `hist_target`, `confusion_matrix`, `parity_fis`,
`nested_boxplots`, `error_by_zone`) y todas las tablas LaTeX del
artículo (`../tables/*.tex`).

---

## 5. Resultados de referencia

Con la semilla fija (42) los cuatro scripts reproducen exactamente los
números del artículo: recall medio de los árboles ≈ 0.85 (límite de
2 mpy) y ≈ 0.96 (límite de 8 mpy); FIS out-of-fold $R^2 \approx 0.7$ y
MAE ≈ 2.5–2.8 mpy con 7–13 reglas por pliegue; regresión lineal
$R^2 = 0.447$; los valores exactos quedan en `results/comparison.csv`.
