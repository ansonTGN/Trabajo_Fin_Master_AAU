# Proyecto Fin de Máster en Ciencia de Datos
## IA generativa aplicada a una aplicación de apoyo a la lactancia materna

> **Documento original:** proyecto desarrollado en el contexto tecnológico de 2023–2024.  
> **Actualización del estado del arte:** 19 de agosto de 2026.  
> **Objetivo de esta revisión:** conservar el planteamiento, la arquitectura experimental y el espíritu del proyecto original, incorporando de forma explícita —y sin anacronismos— la evolución posterior de la inteligencia artificial.

---

## Índice

1. [Nota sobre esta actualización](#1-nota-sobre-esta-actualización)
2. [Proyecto original: contexto y problema](#2-proyecto-original-contexto-y-problema)
3. [Qué era razonable considerar estado del arte en 2023–2024](#3-qué-era-razonable-considerar-estado-del-arte-en-20232024)
4. [Qué ha cambiado entre el proyecto original y 2026](#4-qué-ha-cambiado-entre-el-proyecto-original-y-2026)
5. [Estado del arte de la IA generativa en agosto de 2026](#5-estado-del-arte-de-la-ia-generativa-en-agosto-de-2026)
6. [El ecosistema chino de modelos abiertos y de pesos abiertos](#6-el-ecosistema-chino-de-modelos-abiertos-y-de-pesos-abiertos)
7. [Qué aportan Kimi, DeepSeek, Qwen, GLM y MiniMax](#7-qué-aportan-kimi-deepseek-qwen-glm-y-minimax)
8. [Qué implica esta evolución para el proyecto](#8-qué-implica-esta-evolución-para-el-proyecto)
9. [Arquitectura propuesta desde la perspectiva de 2026](#9-arquitectura-propuesta-desde-la-perspectiva-de-2026)
10. [RAG, grafos de conocimiento y Neo4j](#10-rag-grafos-de-conocimiento-y-neo4j)
11. [Herramientas, agentes y ejecución estructurada](#11-herramientas-agentes-y-ejecución-estructurada)
12. [Multimodalidad](#12-multimodalidad)
13. [Modelos locales, modelos de pesos abiertos y soberanía tecnológica](#13-modelos-locales-modelos-de-pesos-abiertos-y-soberanía-tecnológica)
14. [Cómo seleccionar modelos en 2026](#14-cómo-seleccionar-modelos-en-2026)
15. [Diseño experimental actualizado](#15-diseño-experimental-actualizado)
16. [Evaluación específica del dominio](#16-evaluación-específica-del-dominio)
17. [Seguridad, incertidumbre y supervisión experta](#17-seguridad-incertidumbre-y-supervisión-experta)
18. [Despliegue local y en servidor](#18-despliegue-local-y-en-servidor)
19. [Qué permanece vigente del proyecto original](#19-qué-permanece-vigente-del-proyecto-original)
20. [Qué debe considerarse legado histórico](#20-qué-debe-considerarse-legado-histórico)
21. [Aplicación, Neo4j e imágenes del proyecto original](#21-aplicación-neo4j-e-imágenes-del-proyecto-original)
22. [Conclusiones](#22-conclusiones)
23. [Referencias y fuentes actuales](#23-referencias-y-fuentes-actuales)
24. [Conoce más sobre mí](#24-conoce-más-sobre-mí)

---

# 1. Nota sobre esta actualización

Este repositorio nació como parte de un **Proyecto Fin de Máster en Ciencia de Datos** orientado a estudiar el uso de modelos generativos de lenguaje para mejorar una aplicación existente relacionada con la lactancia materna.

El README original reflejaba correctamente el momento tecnológico en el que se desarrolló el proyecto. En aquella fase, la comparación propuesta se articulaba alrededor de:

- **GPT-4**, como referencia propietaria de alto rendimiento;
- **Llama 2**, como una de las familias de pesos disponibles más relevantes;
- **Mistral 7B**, como modelo compacto que comenzaba a mostrar el potencial de los modelos abiertos o de pesos abiertos;
- ejecución local mediante herramientas como **`llama.cpp`**;
- servicios remotos y APIs;
- una estructura de conocimiento mantenida en **Neo4j**.

Esta actualización **no reescribe retrospectivamente el proyecto** ni atribuye al trabajo original tecnologías que todavía no existían.

Por tanto, a lo largo de este documento se distinguen explícitamente dos planos:

> **Plano histórico:** qué modelos, herramientas y preguntas eran pertinentes durante el proyecto original.

> **Plano de actualización 2026:** cómo reinterpretaría hoy el mismo problema a la luz del estado actual de la inteligencia artificial.

Esta distinción es importante tanto desde una perspectiva académica como metodológica.

Un proyecto tecnológico también es un documento de su tiempo.

---

# 2. Proyecto original: contexto y problema

## Descripción general

La propuesta del proyecto se centró en investigar cómo mejorar una aplicación existente de lactancia materna mediante técnicas generativas de procesamiento de lenguaje.

La aplicación utilizaba un enfoque basado en **árboles de decisión** para ayudar a resolver problemas específicos planteados por las usuarias.

El objetivo era mejorar la identificación de esos problemas y facilitar la localización de información relevante.

El planteamiento original proponía utilizar modelos generativos de lenguaje para:

1. **Resumir e identificar las temáticas principales de las consultas.**
2. **Localizar puntos adecuados en el árbol de preguntas y respuestas de la aplicación.**
3. **Explorar la generación de una conversación interactiva.**
4. **Identificar preguntas relevantes que pudieran ayudar a resolver el problema planteado.**
5. **Asistir al panel de expertas**, manteniendo el conocimiento especializado como referencia.

La hipótesis principal no era sustituir el conocimiento existente, sino analizar si un modelo generativo podía actuar como **capa de comprensión del lenguaje natural** entre la usuaria y una estructura de conocimiento previamente construida.

Esta intuición continúa siendo plenamente relevante.

---

![Imagen de Introducción](/ESQUEMAS_GRAFICOS/ESQUEMA_PROYECTO_10_12_23.png)

---

# 3. Qué era razonable considerar estado del arte en 2023–2024

Para interpretar correctamente el proyecto hay que situarlo en su contexto.

## GPT-4

GPT-4 fue presentado por OpenAI en marzo de 2023 y se convirtió rápidamente en una referencia para tareas complejas de comprensión y generación de lenguaje.

Por ello, utilizarlo como referencia comercial en el proyecto original era coherente.

## Llama 2

Meta publicó Llama 2 en julio de 2023.

La disponibilidad de sus pesos permitió experimentar con una alternativa que podía ejecutarse fuera de una API exclusivamente propietaria.

Esta posibilidad era especialmente relevante para investigar:

- privacidad;
- despliegue local;
- control de infraestructura;
- costes;
- personalización.

## Mistral 7B

Mistral AI presentó Mistral 7B en septiembre de 2023.

Su importancia no estaba únicamente en su rendimiento, sino en demostrar que un modelo relativamente compacto podía ser competitivo y resultar suficientemente pequeño como para facilitar experimentación local.

Por tanto, la combinación:

```text
GPT-4
   vs.
Llama 2
   vs.
Mistral 7B
```

representaba una comparación razonable para ese momento.

## Qué NO formaba parte del proyecto original

No deben presentarse como componentes originales del proyecto tecnologías o modelos posteriores, entre ellos:

- DeepSeek-R1;
- DeepSeek-V4;
- Kimi K2 o Kimi K3;
- Qwen3.x;
- GLM-5;
- MiniMax M3;
- GPT-5.x;
- Claude 5;
- Gemini 3;
- los actuales sistemas de agentes de larga duración;
- los modelos multimodales actuales de frontera;
- el ecosistema moderno de tool calling;
- las arquitecturas modernas de agentes con uso intensivo de herramientas;
- la generalización actual de ventanas de contexto del orden de cientos de miles o un millón de tokens.

Esos desarrollos pertenecen a la **actualización de 2026**, no a la memoria histórica del proyecto.

---

# 4. Qué ha cambiado entre el proyecto original y 2026

La evolución no puede resumirse simplemente como:

> «los modelos actuales son mejores».

Ha cambiado la propia definición práctica de un sistema de IA generativa.

En 2023 era frecuente pensar en:

```text
Prompt
  ↓
LLM
  ↓
Respuesta
```

En 2026 es más apropiado pensar en:

```text
Usuario
  ↓
Comprensión multimodal
  ↓
Clasificación / estructuración
  ↓
Recuperación de conocimiento
  ↓
Razonamiento
  ↓
Uso de herramientas
  ↓
Comprobaciones
  ↓
Respuesta estructurada
  ↓
Evaluación / escalado / supervisión humana
```

## Evolución conceptual

| Proyecto original | Perspectiva 2026 |
|---|---|
| LLM como generador de texto | Modelo como motor dentro de un sistema |
| Chat | Agentes y workflows |
| Prompt → respuesta | Ciclos de razonamiento y herramientas |
| Texto | Texto + imagen + audio + vídeo + documentos |
| Contexto relativamente limitado | Contextos muy extensos |
| Modelo aislado | Modelo + RAG + herramientas + memoria + reglas |
| Respuesta libre | Salidas estructuradas |
| Un modelo | Routing entre modelos |
| Fine-tuning como opción destacada | RAG, herramientas y fine-tuning según necesidad |
| Benchmark general | Evals específicas del dominio |
| API o local | Arquitecturas cloud, locales, edge e híbridas |

La consecuencia principal es que la pregunta de investigación cambia de:

> **¿Qué LLM responde mejor?**

a:

> **¿Qué arquitectura produce el resultado más fiable, útil, seguro, trazable y eficiente para esta tarea concreta?**

---

# 5. Estado del arte de la IA generativa en agosto de 2026

> **Fotografía temporal: 19 de agosto de 2026.**
>
> Los modelos evolucionan con gran rapidez. Los nombres incluidos en esta sección deben entenderse como referencias del momento y no como una lista permanente ni como un ranking absoluto.

## 5.1 Modelos de frontera propietarios

Entre los ecosistemas propietarios relevantes en agosto de 2026 se encuentran, entre otros:

### OpenAI — GPT-5.6

La familia GPT-5.6 está organizada en diferentes niveles de capacidad y coste:

- **GPT-5.6 Sol**;
- **GPT-5.6 Terra**;
- **GPT-5.6 Luna**.

El énfasis ya no está únicamente en generación de texto, sino en:

- razonamiento;
- programación;
- uso de herramientas;
- tareas agentivas;
- ejecución de workflows de varios pasos;
- coordinación de agentes;
- trabajo con documentos y artefactos.

### Anthropic — Claude 5

La oferta actual incluye modelos como:

- **Claude Fable 5**;
- **Claude Opus 5**;
- **Claude Sonnet 5**.

La evolución de Claude es especialmente representativa del desplazamiento desde el chatbot hacia:

- agentes de larga duración;
- razonamiento adaptativo;
- programación;
- uso de herramientas;
- trabajo con grandes contextos;
- workflows empresariales.

### Google — Gemini 3

Google mantiene una familia multimodal con modelos como:

- **Gemini 3.7 Flash**;
- **Gemini 3.1 Pro** (preview en la documentación consultada);
- variantes especializadas de imagen, voz y tiempo real.

Gemini ilustra otro cambio importante: la **multimodalidad deja de ser una característica periférica** y pasa a formar parte del diseño central de los modelos.

---

## 5.2 Modelos abiertos y de pesos abiertos

El estado del arte ya no está formado únicamente por proveedores cerrados.

El ecosistema de modelos cuyos pesos pueden descargarse ha avanzado de manera extraordinaria.

Algunos ejemplos relevantes son:

- Mistral 3;
- Kimi K3;
- DeepSeek V4;
- diferentes familias Qwen;
- GLM-5.x;
- MiniMax M3.

Esto tiene consecuencias técnicas importantes:

- posibilidad de autoalojamiento;
- mayor independencia del proveedor;
- investigación reproducible;
- cuantización;
- fine-tuning;
- ejecución en infraestructuras privadas;
- creación de servicios especializados;
- comparación entre proveedores utilizando una misma arquitectura de aplicación.

Sin embargo:

> **pesos abiertos no significa automáticamente software libre u open source en sentido estricto.**

Siempre debe revisarse la licencia concreta de cada modelo.

## 5.3 Fotografía comparativa del ecosistema — agosto de 2026

La siguiente tabla no pretende establecer un ranking. Su finalidad es mostrar cómo se ha ampliado el espacio de soluciones respecto al proyecto original.

| Ecosistema | Referencia en agosto de 2026 | Distribución | Rasgo relevante para esta actualización |
|---|---|---|---|
| OpenAI | GPT-5.6 Sol / Terra / Luna | API propietaria | razonamiento, herramientas, workflows agentivos y diferentes niveles coste/capacidad |
| Anthropic | Claude Fable 5 / Opus 5 / Sonnet 5 | API propietaria | agentes de larga duración, razonamiento adaptativo, visión y contextos de 1M |
| Google | Gemini 3.7 Flash / Gemini 3.1 Pro | API propietaria | multimodalidad, agentes, tiempo real y especialización por modalidad |
| Mistral AI | Mistral Large 3 / Ministral 3 | Apache 2.0 | modelos abiertos desde edge hasta gran MoE |
| Moonshot AI | Kimi K3 | pesos abiertos, licencia Kimi K3 | 2,8T parámetros totales, multimodalidad nativa, 1M de contexto y orientación agentiva |
| DeepSeek | DeepSeek V4 Pro / Flash | pesos abiertos | MoE, atención dispersa, 1M de contexto y foco en eficiencia |
| Alibaba / Qwen | Qwen3.8-2.4T-A95B / Qwen3.8-Max | pesos abiertos + servicio gestionado | apertura a escala Max, ecosistema amplio y variantes de código/multimodales |
| Z.AI / Zhipu | GLM-5.2 | modelo abierto | contexto 1M, atención dispersa y trabajo agentivo de larga duración |
| MiniMax | MiniMax M3 | pesos abiertos | 1M de contexto, multimodalidad, coding y computer use |

La tabla muestra una diferencia fundamental respecto a 2023:

```text
ANTES
modelo propietario de frontera
        vs.
modelo abierto relativamente pequeño

AHORA
modelo propietario de frontera
        vs.
modelo abierto/de pesos abiertos de frontera
        vs.
modelo local eficiente
        vs.
sistemas híbridos multi-modelo
```

---

# 6. El ecosistema chino de modelos abiertos y de pesos abiertos

Uno de los cambios más importantes desde la versión original de este proyecto ha sido la consolidación de China como uno de los principales centros de desarrollo de modelos avanzados.

En 2023, una comparación simplificada podía centrarse en:

```text
OpenAI
Meta
Mistral
```

En 2026 esa representación sería claramente incompleta.

El ecosistema chino incluye varias líneas de investigación y producto capaces de competir en:

- razonamiento;
- programación;
- agentes;
- multimodalidad;
- contexto largo;
- eficiencia de inferencia;
- modelos Mixture-of-Experts;
- atención dispersa;
- modelos abiertos o de pesos abiertos.

Entre las familias más significativas se encuentran:

```text
Moonshot AI  → Kimi
DeepSeek     → DeepSeek
Alibaba      → Qwen
Z.AI / Zhipu → GLM
MiniMax      → MiniMax
```

Su importancia para este proyecto no reside en su nacionalidad, sino en que **modifican las hipótesis técnicas disponibles**.

Ya no es razonable utilizar:

> «modelo propietario occidental frente a pequeño modelo abierto»

como única estructura comparativa.

Ahora existe una categoría adicional:

> **modelos de gran capacidad y pesos abiertos que compiten en tareas anteriormente reservadas a modelos cerrados de frontera.**

---

# 7. Qué aportan Kimi, DeepSeek, Qwen, GLM y MiniMax

## 7.1 Kimi K3 — Moonshot AI

### Situación en agosto de 2026

Kimi K3 fue presentado en 2026 por Moonshot AI.

Sus pesos completos han sido publicados bajo la **Kimi K3 License**.

Por precisión terminológica, en este documento se describe principalmente como **modelo de pesos abiertos**, aunque Moonshot utiliza también la expresión *open frontier intelligence*.

Entre sus características destacan:

- arquitectura Mixture-of-Experts;
- aproximadamente **2,8 billones de parámetros totales (2.8 trillion en la nomenclatura anglosajona)**;
- razonamiento configurable;
- multimodalidad nativa;
- ventana de contexto de hasta **1 millón de tokens**;
- orientación a tareas agentivas;
- programación de larga duración;
- tool calling;
- salidas estructuradas;
- compatibilidad con motores de inferencia como vLLM y SGLang.

Kimi K3 aplica además cuantización nativa durante el entrenamiento posterior, con pesos MXFP4 y activaciones MXFP8.

### Qué aporta conceptualmente

Kimi K3 representa varias tendencias simultáneas:

1. **Escalado de modelos abiertos o de pesos abiertos hasta dimensiones de frontera.**
2. **Mixture-of-Experts como mecanismo para separar parámetros totales de parámetros realmente activados.**
3. **Contextos del orden de un millón de tokens.**
4. **Razonamiento integrado en el funcionamiento normal del modelo.**
5. **Orientación hacia tareas largas y agentivas.**
6. **Compatibilidad con infraestructuras de inferencia independientes del proveedor.**

### Implicación para este proyecto

Un proyecto como el presente ya no tendría que comparar únicamente:

```text
modelo comercial potente
        vs.
modelo local pequeño
```

Podría incorporar:

```text
modelo propietario de frontera
        vs.
modelo abierto/de pesos abiertos de frontera
        vs.
modelo abierto pequeño y local
```

Es una diferencia metodológica muy importante.

---

## 7.2 DeepSeek V4

DeepSeek constituye probablemente uno de los ejemplos más claros de la aceleración experimentada por el ecosistema abierto.

En abril de 2026, DeepSeek presentó **DeepSeek V4 Preview**, con dos variantes:

- **DeepSeek-V4-Pro**;
- **DeepSeek-V4-Flash**.

Según su documentación oficial:

- V4-Pro utiliza una arquitectura de aproximadamente 1,6T parámetros totales y 49B activos;
- V4-Flash reduce significativamente tamaño y coste;
- ambas variantes soportan contextos de hasta 1M tokens;
- combinan modos de razonamiento y no razonamiento;
- incorporan optimizaciones orientadas a agentes;
- utilizan mecanismos de atención dispersa;
- mantienen APIs compatibles con patrones de uso de OpenAI y Anthropic.

### Qué aporta

DeepSeek ha contribuido especialmente a hacer visibles tres ideas:

#### 1. Eficiencia

No toda mejora necesita obtenerse activando todos los parámetros del modelo.

Las arquitecturas MoE y los mecanismos de atención dispersa permiten buscar una relación más favorable entre:

```text
capacidad
coste
memoria
latencia
```

#### 2. Razonamiento abierto

El razonamiento avanzado deja de ser exclusivamente una propiedad de modelos cerrados.

#### 3. Compatibilidad

La compatibilidad con APIs ampliamente utilizadas reduce la dependencia del proveedor.

Para una aplicación experimental, esto permite cambiar de backend manteniendo una parte significativa del código.

---

## 7.3 Qwen — Alibaba

Qwen es probablemente uno de los ecosistemas más amplios.

Es importante distinguir entre:

- modelos propietarios servidos por Alibaba;
- modelos cuyos pesos han sido publicados;
- modelos generales;
- modelos especializados en código, visión, audio, imagen, embeddings o agentes.

### Evolución

Qwen3 abrió numerosos modelos bajo Apache 2.0, incluyendo arquitecturas densas y MoE.

Posteriormente aparecieron líneas como:

- Qwen3-Coder;
- Qwen3.5;
- Qwen3.6;
- Qwen3-Coder-Next;
- Qwen Code;
- modelos específicos de embeddings y reranking;
- modelos multimodales.

En agosto de 2026, **Qwen3.8-Max** es la referencia de máxima escala de la familia y constituye además un cambio importante en la estrategia de apertura de Qwen.

Qwen presentó Qwen3.8-Max el 2 de agosto de 2026 como su **primer modelo open-weight a escala Max**. El checkpoint abierto `Qwen3.8-2.4T-A95B` alcanza aproximadamente 2,4 billones de parámetros totales y 95.000 millones de parámetros activos. Sus pesos están publicados bajo una licencia específica **Qwen3.8-Max License**.

Conviene, sin embargo, distinguir el checkpoint abierto de la versión gestionada del servicio Qwen3.8-Max. La documentación de Qwen indica que el servicio oficial añade capacidades adicionales —por ejemplo determinadas funciones de visión, contexto de 1M y herramientas integradas— sobre la base del modelo abierto.

Esta precisión vuelve a ilustrar una regla importante:

> **“modelo de pesos abiertos”, “servicio gestionado” y “producto comercial” pueden ser capas diferentes de una misma familia.**

Y también:

> **Qwen no es un único modelo ni toda la familia comparte necesariamente la misma licencia o modalidad de distribución.**

### Qué aporta

Qwen representa especialmente bien la idea de **ecosistema completo**:

```text
LLM general
   +
modelo de código
   +
modelo multimodal
   +
embeddings
   +
reranking
   +
agentes
   +
herramientas
   +
CLI
```

Para este proyecto, esto implica que ya no es obligatorio recurrir a proveedores diferentes para cada componente.

Un mismo ecosistema puede proporcionar varias piezas de una arquitectura RAG o agentiva.

---

## 7.4 GLM-5.x — Z.AI / Zhipu

La familia GLM ha evolucionado hacia lo que Z.AI denomina **Agentic Engineering**.

GLM-5 y sus sucesores se orientan especialmente a:

- programación;
- terminal;
- ingeniería de software;
- workflows de larga duración;
- uso reiterado de herramientas;
- contextos muy largos.

GLM-5.2 proporciona un contexto de hasta 1M tokens y utiliza mejoras de atención dispersa para reducir el coste computacional del contexto largo.

### Qué aporta

GLM muestra que la evaluación moderna de modelos está desplazándose desde:

```text
pregunta → respuesta
```

hacia:

```text
objetivo
  ↓
plan
  ↓
herramienta
  ↓
resultado
  ↓
revisión
  ↓
nuevo paso
  ↓
resultado final
```

Es decir, se empieza a evaluar cuánto tiempo puede **mantenerse productivo un modelo dentro de una tarea real**.

Este cambio es especialmente importante para aplicaciones complejas.

---

## 7.5 MiniMax M3

MiniMax M3 combina:

- contexto de hasta 1M tokens;
- multimodalidad;
- programación;
- trabajo agentivo;
- interacción con ordenador;
- una arquitectura de atención dispersa denominada MiniMax Sparse Attention.

MiniMax lo presenta como un modelo de pesos abiertos capaz de reunir en una misma arquitectura:

```text
contexto largo
+
multimodalidad
+
agentes
```

### Qué aporta

MiniMax ilustra otro desplazamiento fundamental:

> la frontera entre «modelo de lenguaje», «modelo multimodal» y «agente» se está difuminando.

El modelo empieza a diseñarse desde el origen para interactuar con:

- interfaces;
- documentos;
- imágenes;
- vídeo;
- aplicaciones;
- herramientas.

---

## 7.6 MiniMax H3 y la convergencia multimodal

MiniMax H3, publicado y posteriormente abierto en 2026, lleva esta tendencia más lejos en el dominio audiovisual.

Integra comprensión de:

- texto;
- imagen;
- vídeo;
- audio;

y generación audiovisual.

No es un sustituto directo del LLM central que utilizaría este proyecto, pero demuestra una tendencia relevante:

> **la IA generativa está convergiendo hacia modelos capaces de operar sobre múltiples modalidades en una arquitectura común.**

---

# 8. Qué implica esta evolución para el proyecto

La aparición de estos modelos no obliga a rehacer conceptualmente el proyecto.

Al contrario: **refuerza algunas de sus ideas originales**.

## 8.1 La comparación debe ser arquitectónica

En 2023 podía ser razonable preguntar:

> ¿GPT-4 o un modelo abierto?

En 2026 la comparación adecuada sería:

```text
ARQUITECTURA A
Modelo propietario de frontera + RAG

ARQUITECTURA B
Modelo eficiente por API + RAG

ARQUITECTURA C
Modelo open-weight de gran capacidad + RAG

ARQUITECTURA D
Modelo local pequeño + grafo + reglas

ARQUITECTURA E
Routing dinámico entre varios modelos
```

El objeto experimental deja de ser únicamente el modelo.

Pasa a ser **el sistema**.

---

## 8.2 Menor dependencia de un único proveedor

La existencia de varios modelos competitivos permite diseñar interfaces abstractas.

Ejemplo:

```text
                 ┌──────────────────┐
                 │     APLICACIÓN   │
                 └────────┬─────────┘
                          │
                    Model Gateway
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
        ▼                 ▼                 ▼
      OpenAI            Kimi            DeepSeek
        │                 │                 │
        └─────────┬───────┴───────┬─────────┘
                  ▼               ▼
                Qwen             Local
```

La aplicación puede elegir un modelo según:

- coste;
- sensibilidad de los datos;
- dificultad;
- latencia;
- disponibilidad;
- capacidad requerida.

---

## 8.3 La apertura ya no equivale a baja capacidad

Una de las hipótesis implícitas habituales en 2023 era:

```text
modelo cerrado
      ≈
máxima capacidad

modelo abierto
      ≈
más control pero menor rendimiento
```

Esa separación se ha debilitado significativamente.

En 2026 existen modelos de pesos abiertos diseñados expresamente para competir en tareas de:

- razonamiento;
- programación;
- agentes;
- contexto largo;
- multimodalidad.

Esto no significa que sean superiores en todas las tareas.

Significa que **el carácter abierto o cerrado ya no permite predecir por sí solo la capacidad del sistema**.

---

## 8.4 Modelos gigantes no significan ejecución doméstica

Hay que evitar otra conclusión incorrecta.

Que un modelo tenga pesos disponibles **no significa que pueda ejecutarse cómodamente en un ordenador personal**.

Por ejemplo, un modelo con centenares de miles de millones o billones de parámetros puede requerir:

- múltiples GPUs;
- grandes cantidades de memoria;
- infraestructura especializada;
- cuantización agresiva;
- inferencia distribuida.

Por ello deben distinguirse dos conceptos:

```text
PESOS DISPONIBLES
        ≠
EJECUCIÓN LOCAL PRÁCTICA
```

Para un laboratorio doméstico o un equipo convencional siguen siendo muy relevantes los modelos pequeños y medianos.

---

## 8.5 Implicación estratégica: IA multipolar

Desde una perspectiva tecnológica, la aparición de Kimi, DeepSeek, Qwen, GLM y MiniMax muestra que la innovación de frontera ya no puede analizarse exclusivamente a través de empresas estadounidenses.

Esto tiene varias implicaciones:

- mayor diversidad de arquitecturas;
- competencia en coste;
- competencia en eficiencia;
- disponibilidad de pesos;
- nuevas técnicas de atención;
- mayor velocidad de difusión de innovaciones;
- menor capacidad de un único proveedor para definir de forma exclusiva los estándares de facto.

Para proyectos de investigación, esto es positivo porque amplía el espacio experimental.

Al mismo tiempo, exige prestar mayor atención a:

- licencias;
- procedencia del modelo;
- política de datos;
- infraestructura;
- mantenimiento;
- compatibilidad;
- reproducibilidad;
- seguridad de la cadena de suministro.

---

# 9. Arquitectura propuesta desde la perspectiva de 2026

La arquitectura que hoy consideraría más coherente con la idea original sería híbrida.

```text
                              ┌─────────────────────┐
                              │       USUARIA       │
                              └──────────┬──────────┘
                                         │
                                         ▼
                            ┌────────────────────────┐
                            │ Interfaz conversacional│
                            └───────────┬────────────┘
                                        │
                                        ▼
                         ┌──────────────────────────────┐
                         │ Comprensión de la consulta   │
                         │ + extracción estructurada    │
                         └──────────────┬───────────────┘
                                        │
                ┌───────────────────────┼───────────────────────┐
                │                       │                       │
                ▼                       ▼                       ▼
       ┌────────────────┐      ┌─────────────────┐      ┌─────────────────┐
       │ Neo4j / Grafo  │      │ RAG documental │      │ Reglas          │
       │ de conocimiento│      │ / Vector Search │      │ deterministas  │
       └───────┬────────┘      └────────┬────────┘      └────────┬────────┘
               │                        │                        │
               └────────────────────────┼────────────────────────┘
                                        │
                                        ▼
                           ┌─────────────────────────┐
                           │ Modelo / Router de IA   │
                           └────────────┬────────────┘
                                        │
                        ┌───────────────┼────────────────┐
                        │               │                │
                        ▼               ▼                ▼
                 modelo local      API eficiente    frontier model
                        │               │                │
                        └───────────────┼────────────────┘
                                        │
                                        ▼
                          ┌──────────────────────────┐
                          │ Validación / guardrails │
                          └────────────┬─────────────┘
                                       │
                         ┌─────────────┴─────────────┐
                         │                           │
                         ▼                           ▼
                  Respuesta segura            Panel de expertas
```

La principal idea es:

> **el LLM no sustituye el conocimiento experto ni la arquitectura de la aplicación; se integra dentro de ella.**

---

# 10. RAG, grafos de conocimiento y Neo4j

## 10.1 Retrieval-Augmented Generation

La consolidación de **RAG — Retrieval-Augmented Generation** constituye una de las evoluciones más relevantes para este proyecto.

En vez de esperar que el modelo contenga internamente todo el conocimiento necesario:

```text
consulta
   ↓
búsqueda
   ↓
documentos relevantes
   ↓
modelo
   ↓
respuesta fundamentada
```

Esto permite separar:

```text
CONOCIMIENTO
      de
CAPACIDAD LINGÜÍSTICA
```

## 10.2 Por qué encaja especialmente bien

En una aplicación basada en conocimiento experto, el modelo no debería ser la única fuente de verdad.

Una arquitectura RAG permite:

- actualizar información sin reentrenar el modelo;
- conocer qué documentos se recuperaron;
- limitar la respuesta a fuentes seleccionadas;
- citar evidencia;
- auditar errores;
- comparar diferentes modelos utilizando exactamente el mismo corpus.

---

## 10.3 Knowledge Graph + RAG

El uso de Neo4j que formaba parte del proyecto original adquiere todavía más interés desde la perspectiva actual.

Un grafo puede representar relaciones explícitas:

```text
problema
   ↓
síntoma
   ↓
pregunta
   ↓
condición
   ↓
nodo
   ↓
recomendación
```

Mientras que la búsqueda vectorial permite relaciones semánticas:

```text
consulta textual
      ↓
embedding
      ↓
similitud
      ↓
fragmentos relacionados
```

La combinación puede ser:

```text
              CONSULTA
                 │
        ┌────────┴────────┐
        │                 │
        ▼                 ▼
 Vector Search         Neo4j
 semántico              grafo
        │                 │
        └────────┬────────┘
                 ▼
         contexto recuperado
                 │
                 ▼
                LLM
```

Este enfoque híbrido puede ser más robusto que sustituir completamente el árbol existente.

---

# 11. Herramientas, agentes y ejecución estructurada

## 11.1 Tool Calling

Los modelos actuales pueden recibir definiciones de funciones.

Ejemplo conceptual:

```text
buscar_nodo_grafo()
buscar_documentos()
obtener_preguntas_relacionadas()
validar_regla()
solicitar_revision_experta()
```

El modelo puede decidir utilizar una herramienta cuando lo necesita.

Esto reduce la necesidad de introducir todo el conocimiento en el prompt.

---

## 11.2 De chatbot a agente

Un agente puede realizar ciclos como:

```text
observar
   ↓
razonar
   ↓
seleccionar herramienta
   ↓
ejecutar
   ↓
observar resultado
   ↓
revisar
   ↓
continuar o responder
```

Para este proyecto, sin embargo, la autonomía no debería convertirse en un objetivo por sí misma.

La pregunta correcta es:

> **¿Qué pasos se benefician realmente de comportamiento agentivo y cuáles deben permanecer deterministas?**

En un dominio sensible, suele ser preferible:

```text
máxima autonomía
        ✗

autonomía limitada y evaluada
        ✓
```

---

## 11.3 Salidas estructuradas

En lugar de recibir texto arbitrario, el modelo puede generar una estructura:

```json
{
  "tema": "dolor",
  "subtema": "pendiente_de_clasificar",
  "nodo_candidato": "NODO_123",
  "informacion_faltante": [
    "duracion",
    "localizacion"
  ],
  "confianza": 0.72,
  "requiere_revision_experta": true
}
```

Esto facilita:

- validación;
- tests;
- integración;
- auditoría;
- almacenamiento;
- comparación entre modelos.

Es una de las diferencias prácticas más importantes respecto a utilizar un chatbot como componente aislado.

---

# 12. Multimodalidad

En el proyecto original, el foco estaba principalmente en texto.

En 2026 el estado del arte es multimodal.

Dependiendo del modelo, una única arquitectura puede comprender:

- texto;
- imágenes;
- audio;
- vídeo;
- PDFs;
- interfaces gráficas.

Para este proyecto podrían explorarse en el futuro casos como:

```text
consulta escrita
+
documento
+
imagen
+
audio transcrito
```

Sin embargo, que un modelo pueda procesar una modalidad **no implica que deba utilizarse automáticamente**.

Cada nuevo tipo de dato introduce nuevas cuestiones de:

- privacidad;
- calidad;
- seguridad;
- interpretación;
- consentimiento;
- evaluación.

---

# 13. Modelos locales, modelos de pesos abiertos y soberanía tecnológica

## 13.1 Tres categorías diferentes

Conviene distinguir:

### API propietaria

El modelo se consume como servicio.

Ventajas:

- simplicidad;
- escalabilidad;
- acceso inmediato a modelos de frontera.

Inconvenientes potenciales:

- dependencia del proveedor;
- coste variable;
- política de datos;
- cambios de modelos;
- deprecaciones.

---

### Modelo de pesos abiertos autoalojado

Los pesos pueden desplegarse en infraestructura propia.

Ventajas:

- control;
- privacidad;
- reproducibilidad;
- personalización;
- independencia de API.

Inconvenientes:

- infraestructura;
- administración;
- coste de GPU;
- optimización;
- seguridad;
- actualizaciones.

---

### Modelo local pequeño o mediano

Puede ejecutarse en workstation, servidor pequeño o dispositivo edge.

Ventajas:

- privacidad;
- baja latencia;
- funcionamiento offline;
- coste marginal bajo.

Inconvenientes:

- menor capacidad en tareas complejas;
- limitación de contexto efectivo;
- menor robustez potencial.

---

## 13.2 Arquitectura híbrida

La solución óptima puede combinar varios niveles:

```text
consulta sencilla
      ↓
modelo local pequeño

consulta intermedia
      ↓
modelo eficiente

consulta compleja
      ↓
modelo de frontera

caso sensible / incierto
      ↓
experta humana
```

Esto se denomina frecuentemente **model routing**.

Para una aplicación real puede resultar más eficiente que utilizar siempre el modelo más grande.

---

# 14. Cómo seleccionar modelos en 2026

La versión original del README mencionaba métricas como la **perplejidad**.

La perplejidad sigue siendo una medida útil para investigación sobre modelos de lenguaje.

Pero no es suficiente para seleccionar un sistema aplicado.

## Criterios actuales

### 1. Calidad específica de la tarea

¿Resuelve correctamente *nuestro* problema?

### 2. Groundedness

¿La respuesta está sustentada por información recuperada?

### 3. Hallucination rate

¿Introduce información que no aparece en las fuentes?

### 4. Tool-use accuracy

¿Selecciona y utiliza correctamente las herramientas?

### 5. Structured-output reliability

¿Respeta el esquema exigido?

### 6. Context handling

¿Utiliza adecuadamente información dispersa en contextos largos?

### 7. Latencia

¿Cuánto tarda?

### 8. Coste

¿Cuánto cuesta cada tarea completa?

### 9. Privacidad

¿Dónde se procesan los datos?

### 10. Licencia

¿Puede utilizarse para el objetivo previsto?

### 11. Reproducibilidad

¿Podemos repetir el experimento?

### 12. Capacidad de ejecución local

¿Qué hardware requiere?

### 13. Robustez multilingüe

¿Mantiene calidad en español y otras lenguas relevantes?

### 14. Calibración de incertidumbre

¿Sabe expresar cuándo no dispone de información suficiente?

---

# 15. Diseño experimental actualizado

Si este proyecto comenzara hoy, mantendría la comparación original, pero ampliaría el diseño.

## Grupo A — Modelo propietario de frontera

Objetivo:

> establecer una referencia superior de capacidad.

Ejemplos posibles en agosto de 2026:

- GPT-5.6;
- Claude 5;
- Gemini 3.

---

## Grupo B — Modelo de pesos abiertos de alta capacidad

Objetivo:

> comprobar cuánto de la capacidad de frontera puede obtenerse con mayor control de infraestructura.

Ejemplos:

- Kimi K3;
- DeepSeek V4;
- GLM-5.x;
- MiniMax M3;
- variantes Qwen abiertas.

---

## Grupo C — Modelo local

Objetivo:

> evaluar privacidad, coste, independencia y viabilidad offline.

Podrían seleccionarse modelos pequeños o medianos de:

- Qwen;
- Mistral;
- otras familias compatibles con `llama.cpp`, Ollama, vLLM o SGLang.

---

## Grupo D — Baseline determinista

Es fundamental conservar un sistema de referencia sin LLM:

```text
árbol original
+
reglas
+
búsqueda convencional
```

Sin baseline no sabemos si la IA realmente mejora el sistema.

---

# 16. Evaluación específica del dominio

La mejor mejora metodológica respecto al proyecto original sería construir un **conjunto de evaluación propio**.

## Estructura sugerida de cada caso

```yaml
id: CASO_001

consulta:
  texto: "..."

clasificacion_esperada:
  tema: "..."
  subtema: "..."

nodo_correcto:
  id: "..."

informacion_relevante:
  - "..."

informacion_faltante:
  - "..."

preguntas_esperables:
  - "..."

respuesta_base:
  - "..."

criterios_de_seguridad:
  - "..."

requiere_revision_experta: true
```

## Métricas

| Métrica | Pregunta |
|---|---|
| Exactitud temática | ¿Clasifica correctamente el problema? |
| Node accuracy | ¿Localiza el nodo correcto? |
| Recall | ¿Detecta todos los elementos importantes? |
| Precisión | ¿Evita inventar elementos? |
| Groundedness | ¿Se apoya en las fuentes recuperadas? |
| Question quality | ¿Pregunta lo que realmente falta? |
| Escalado | ¿Deriva correctamente los casos dudosos? |
| Consistencia | ¿Casos equivalentes producen resultados equivalentes? |
| Tool accuracy | ¿Utiliza correctamente las herramientas? |
| Latencia | ¿Cuánto tarda? |
| Coste | ¿Qué coste tiene la consulta completa? |

---

## Evals antes que rankings

Los rankings generales son útiles para seleccionar candidatos.

Pero no deberían determinar directamente qué modelo se utiliza.

El criterio final debería ser:

```text
benchmark general
       ↓
selección de candidatos
       ↓
benchmark propio
       ↓
pruebas de seguridad
       ↓
coste / latencia
       ↓
decisión
```

---

# 17. Seguridad, incertidumbre y supervisión experta

Este proyecto se sitúa en un dominio relacionado con salud.

Por ello hay una diferencia esencial entre:

```text
generar una respuesta plausible
```

y:

```text
generar una respuesta suficientemente fiable
para incorporarse a un sistema real
```

## Principios

### El modelo no debe ser la fuente única de conocimiento

La información principal debería proceder de:

- conocimiento validado;
- árbol;
- grafo;
- documentación;
- reglas.

### Debe existir una vía de escalado

Ejemplo:

```text
confianza alta
     ↓
respuesta basada en conocimiento validado

confianza media
     ↓
pregunta adicional

confianza baja
     ↓
revisión experta
```

### La incertidumbre es una salida válida

El sistema debe poder producir:

```text
"No dispongo de información suficiente para determinarlo."
```

en lugar de completar las lagunas mediante generación probabilística.

---

## El panel de expertas sigue siendo central

Uno de los elementos más acertados del planteamiento original era evaluar el LLM como **asistente del panel experto**.

La evolución de los modelos no elimina esa necesidad.

Al contrario, cuanto mayores son las capacidades, más importante es disponer de:

- gobernanza;
- evaluación;
- trazabilidad;
- límites de autonomía;
- validación humana cuando sea necesaria.

---

# 18. Despliegue local y en servidor

El README original describía `llama.cpp` de forma detallada mediante parámetros concretos de línea de comandos.

Esa información tenía sentido como fotografía técnica del momento.

En 2026 es preferible enlazar a la documentación oficial, porque los parámetros cambian continuamente.

## 18.1 llama.cpp

`llama.cpp` continúa siendo uno de los proyectos fundamentales para inferencia local.

Resulta especialmente útil para:

- modelos GGUF;
- cuantización;
- CPU;
- GPU;
- equipos personales;
- servidores ligeros.

Proyecto:

[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)

---

## 18.2 Ollama

Ollama simplifica el uso de modelos locales y permite trabajar con:

- chat;
- embeddings;
- modelos multimodales compatibles;
- tool calling;
- APIs;
- ejecución local.

Documentación:

[Ollama](https://docs.ollama.com/)

---

## 18.3 vLLM

vLLM está orientado especialmente a servir modelos con alto rendimiento.

Resulta más apropiado cuando se necesita:

- concurrencia;
- batching;
- GPUs;
- serving;
- APIs compatibles.

Documentación:

[vLLM](https://docs.vllm.ai/)

---

## 18.4 SGLang

SGLang se ha consolidado como otra infraestructura relevante para inferencia y workloads agentivos.

Proyecto:

[SGLang](https://github.com/sgl-project/sglang)

---

## 18.5 Elección práctica

```text
PORTÁTIL / WORKSTATION
        ↓
llama.cpp / Ollama

SERVIDOR GPU
        ↓
vLLM / SGLang

FRONTERA SIN INFRAESTRUCTURA PROPIA
        ↓
API

ENTORNO SENSIBLE
        ↓
modelo autoalojado
```

---

# 19. Qué permanece vigente del proyecto original

A pesar del enorme cambio tecnológico, varias intuiciones originales han envejecido bien.

## 1. Utilizar IA para interpretar lenguaje natural

Sigue siendo una de las aplicaciones más naturales de los modelos.

## 2. No sustituir automáticamente el árbol

La estructura determinista continúa teniendo valor.

## 3. Combinar IA con conocimiento explícito

Actualmente esta idea encaja directamente con:

- RAG;
- knowledge graphs;
- tool calling.

## 4. Comparar soluciones comerciales y autoalojadas

Esta pregunta es incluso más importante ahora.

## 5. Mantener al experto dentro del sistema

La IA puede actuar como:

```text
asistente
clasificador
navegador
recuperador
resumidor
generador de preguntas
```

sin convertirse necesariamente en autoridad final.

## 6. Evaluar experimentalmente

La pregunta central continúa siendo:

> **¿La incorporación de IA mejora realmente el sistema?**

---

# 20. Qué debe considerarse legado histórico

Algunas partes del README original deben conservarse como documentación histórica, pero no como recomendaciones actuales.

## Modelos concretos

- Llama 2;
- Mistral 7B;
- GPT-4 como referencia de frontera.

Son importantes para comprender el contexto del proyecto, pero no representan el estado del arte de 2026.

## Leaderboards antiguos

Los rankings de modelos cambian demasiado rápidamente.

No deberían considerarse fuentes permanentes.

## Catálogos de modelos de TheBloke

TheBloke fue extremadamente importante para el ecosistema temprano de cuantización y modelos GGUF.

Sin embargo, hoy el ecosistema es mucho más amplio y numerosos proveedores publican directamente formatos optimizados o integraciones oficiales.

## Parámetros concretos de servidores antiguos

Los flags de `llama.cpp` o de `llama-cpp-python` deben consultarse en la documentación correspondiente a la versión utilizada.

---

# 21. Aplicación, Neo4j e imágenes del proyecto original

## Docker + Neo4j

La arquitectura original utilizaba un servicio Docker con Neo4j para mantener la estructura de datos de la aplicación.

![Esquema Neo4jDocker](/ESQUEMAS_GRAFICOS/NEOEsquema1.png)

Desde la perspectiva de 2026, este componente sigue siendo conceptualmente válido.

Una extensión natural sería:

```text
Neo4j
+
Vector Database / Vector Index
+
RAG
+
LLM
+
reglas
```

---

## Aplicación web

Esquema conceptual original:

![Esquema conceptual](/ESQUEMAS_GRAFICOS/APLICATIVO_WEB.png)

Imagen de la aplicación construida:

![imagen WEB](/ESQUEMAS_GRAFICOS/WEB.png)

**Nota:** para el funcionamiento de la aplicación original es necesario mantener disponible el servicio Docker con Neo4j que contiene la estructura utilizada por el proyecto.

El *dump* de esos datos no se proporciona por motivos de confidencialidad.

---

# 22. Conclusiones

Cuando se desarrolló este proyecto, la pregunta técnica podía formularse aproximadamente así:

> **¿Puede un gran modelo de lenguaje mejorar la comprensión de las consultas y ayudar a navegar un árbol de decisión?**

La pregunta sigue siendo válida.

Pero el estado del arte de 2026 permite formular una versión mucho más potente:

> **¿Cómo podemos combinar modelos generativos, razonamiento, recuperación de información, grafos, reglas, herramientas y supervisión humana para construir un sistema más eficaz, trazable y seguro?**

Ese cambio resume la evolución de los últimos años.

## De modelo a sistema

La unidad de innovación ya no es exclusivamente el LLM.

Es la arquitectura completa:

```text
MODELO
  +
DATOS
  +
RAG
  +
GRAFO
  +
REGLAS
  +
TOOLS
  +
EVALS
  +
SUPERVISIÓN
```

## De un mercado concentrado a un ecosistema multipolar

La aparición de Kimi, DeepSeek, Qwen, GLM y MiniMax modifica además la estructura del campo.

Los modelos avanzados de pesos abiertos ya no son únicamente alternativas pequeñas.

Algunos intentan competir directamente con modelos propietarios de frontera.

Esto introduce:

- más opciones;
- mayor presión sobre costes;
- nuevas arquitecturas;
- nuevas licencias;
- más posibilidades de autoalojamiento;
- menor dependencia de un único proveedor.

Pero también obliga a ser más rigurosos.

Un modelo no debe seleccionarse por:

- nacionalidad;
- notoriedad;
- tamaño;
- posición en un ranking;
- etiqueta «open»;
- marketing.

Debe seleccionarse mediante **evaluación reproducible sobre la tarea concreta**.

---

## La intuición fundamental del proyecto permanece

La tecnología ha cambiado.

Los modelos han cambiado.

Las herramientas han cambiado.

Pero la idea central continúa siendo útil:

> **usar la inteligencia artificial como una capa que ayude a comprender mejor las preguntas de las personas y conectarlas con conocimiento útil, manteniendo la capacidad de evaluación, supervisión y control.**

Desde esa perspectiva, el proyecto original no queda invalidado por la evolución de la IA.

Se convierte en el punto de partida de una arquitectura mucho más amplia.

---

# 23. Referencias y fuentes actuales

> Se priorizan fuentes oficiales de los desarrolladores de cada tecnología.  
> Esta sección debe actualizarse periódicamente porque el ecosistema cambia con rapidez.

## Contexto histórico

- OpenAI — GPT-4  
  [https://openai.com/research/gpt-4](https://openai.com/research/gpt-4)

- Meta — Llama  
  [https://www.llama.com/](https://www.llama.com/)

- Mistral AI — Mistral 7B  
  [https://mistral.ai/news/announcing-mistral-7b/](https://mistral.ai/news/announcing-mistral-7b/)

## OpenAI

- GPT-5.6  
  [https://openai.com/index/gpt-5-6/](https://openai.com/index/gpt-5-6/)

## Anthropic

- Claude model overview  
  [https://platform.claude.com/docs/en/about-claude/models/overview](https://platform.claude.com/docs/en/about-claude/models/overview)

## Google

- Gemini API models  
  [https://ai.google.dev/gemini-api/docs/models](https://ai.google.dev/gemini-api/docs/models)

## Mistral

- Mistral 3  
  [https://mistral.ai/news/mistral-3/](https://mistral.ai/news/mistral-3/)

## Moonshot AI / Kimi

- Kimi K3 — documentación  
  [https://platform.moonshot.ai/docs/guide/kimi-k3-quickstart](https://platform.moonshot.ai/docs/guide/kimi-k3-quickstart)

- Kimi K3 — repositorio y pesos  
  [https://github.com/MoonshotAI/Kimi-K3](https://github.com/MoonshotAI/Kimi-K3)

## DeepSeek

- DeepSeek V4  
  [https://api-docs.deepseek.com/news/news260424/](https://api-docs.deepseek.com/news/news260424/)

- DeepSeek API  
  [https://api-docs.deepseek.com/](https://api-docs.deepseek.com/)

## Qwen

- Qwen  
  [https://qwen.ai/](https://qwen.ai/)

- Qwen3  
  [https://qwenlm.github.io/blog/qwen3/](https://qwenlm.github.io/blog/qwen3/)

- Qwen3.8-Max  
  [https://qwen.ai/blog?id=qwen3.8](https://qwen.ai/blog?id=qwen3.8)

- Qwen3.8-2.4T-A95B — pesos  
  [https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B)

- Qwen3-Coder  
  [https://qwenlm.github.io/blog/qwen3-coder/](https://qwenlm.github.io/blog/qwen3-coder/)

- Qwen Code  
  [https://qwenlm.github.io/qwen-code-docs/](https://qwenlm.github.io/qwen-code-docs/)

## Z.AI / GLM

- GLM-5  
  [https://github.com/zai-org/GLM-5](https://github.com/zai-org/GLM-5)

- GLM-5 documentation  
  [https://docs.z.ai/guides/llm/glm-5](https://docs.z.ai/guides/llm/glm-5)

## MiniMax

- MiniMax M3  
  [https://www.minimax.io/blog/minimax-m3](https://www.minimax.io/blog/minimax-m3)

- MiniMax H3  
  [https://www.minimax.io/blog/minimax-h3](https://www.minimax.io/blog/minimax-h3)

## Inferencia local

- llama.cpp  
  [https://github.com/ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)

- Ollama  
  [https://docs.ollama.com/](https://docs.ollama.com/)

- vLLM  
  [https://docs.vllm.ai/](https://docs.vllm.ai/)

- SGLang  
  [https://github.com/sgl-project/sglang](https://github.com/sgl-project/sglang)

## Neo4j

- Neo4j  
  [https://neo4j.com/](https://neo4j.com/)

---

# 24. Conoce más sobre mí

Te invito a ver el siguiente vídeo para conocer un poco más sobre mí y la visión que guía mi trabajo:

[![Video de Presentación](https://img.youtube.com/vi/0CUdsXlIllE/0.jpg)](https://youtu.be/0CUdsXlIllE?si=mFSiEfiN4bOmdJkA)

---

## Nota final de mantenimiento

Este README contiene dos tipos de información:

```text
CONTENIDO HISTÓRICO DEL PROYECTO
              +
ACTUALIZACIÓN DEL ESTADO DEL ARTE
```

Para evitar futuros anacronismos, cualquier nueva revisión debería mantener explícita esta separación.

Una actualización futura no debería modificar retrospectivamente qué modelos existían cuando se realizó el proyecto.

Debería añadir una nueva capa temporal:

```text
Proyecto original
      ↓
Actualización 2026
      ↓
Actualización futura
```

De esta forma, el repositorio puede funcionar simultáneamente como:

- documentación del proyecto académico;
- registro de la evolución tecnológica;
- laboratorio de comparación;
- punto de partida para nuevas investigaciones.
