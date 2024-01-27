# Proyecto Fin de Master en Ciencia de Datos

## Introducción

# Descripción general del problema

La propuesta de mi tesis de fin de máster se centra en investigar como mejorar la aplicación existente de lactancia materna [^1] mediante el uso de técnicas generativas de procesamiento de lenguaje. Esta aplicación actualmente utiliza un enfoque basado en arboles de decisión [^2] para solucionar los problemas específicos que las usuarias desean resolver en relación con la lactancia materna.

Mi objetivo principal es optimizar este proceso de identificación de los problemas planteados y permitir la búsqueda de soluciones de una manera más eficiente y efectiva. Para lograr esto, propongo la implementación de *modelos generativos de lenguaje* (Ver: Modelos Generativos de Lenguaje) para la extracción de información y resumen extractivo de consultas de usuarias. Esto permitirá:

1. Resumir e identificar las temáticas principales de las consultas.
2. Localizar puntos óptimos en el árbol de preguntas y respuestas de la aplicación.
3. Generar un chat interactivo para la evaluación de dos aspectos clave:
    - Viabilidad de generación de conversaciones automáticas.
    - Identificación de preguntas relevantes para resolver las problemáticas planteadas con el objetivo de asistir al panel de expertas.

Para evaluar la efectividad de estos modelos generativos, intentaré llevar a cabo un estudio comparativo utilizando dos enfoques diferentes. Uno se realizara utilizando el modelo de lenguaje GPT-4 [^3] de OpenAI, ampliamente reconocido por su capacidad de generación de texto, y el segundo será explorar la posible utilización de modelos de lenguaje libres [^4] [^5] para comparar su rendimiento con las soluciones comerciales.

Este proyecto de tesis tiene como objetivo brindar a las madres que utilizan la aplicación una experiencia más enriquecedora al proporcionar *respuestas y soluciones más precisas y útiles* para sus problemas de lactancia materna. Además, se espera que los resultados del estudio comparativo ayuden a identificar la eficacia relativa de los modelos de lenguaje utilizados y contribuyan al avance en el desarrollo de las mejoras e investigaciones en este campo.

[^1]: Referencia a "LACTAPP".
[^2]: Referencia a "arbol-decision".
[^3]: Referencia a "openai2023gpt4".
[^4]: Referencia a "touvron2023llama".
[^5]: Referencia a "Mistral".

---

![Imagen de Introducción](/ESQUEMAS_GRAFICOS/ESQUEMA_PROYECTO_10_12_23.png)




# Uso de Modelos de Lenguaje de Gran Tamaño (LLMs)

Este apartado proporciona una guía sencilla para entender, obtener y utilizar Modelos de Lenguaje de Gran Tamaño (LLMs).

## 1. Obtener y Usar Modelos de Lenguaje de Gran Escala (LLM)

[Tabla Comparativa de Modelos Open LLMs](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)

Los modelos de lenguaje de gran escala se pueden implementar y acceder de diversas maneras, incluyendo:

### Autoalojamiento
- **Descripción**: Utilizar hardware local para realizar inferencias.
- **Ejemplo**: Ejecutar Llama 2 en tu PC usando [`llama.cpp`](https://github.com/ggerganov/llama.cpp).
- **Ventajas**: Ideal para privacidad/seguridad o si ya cuentas con una GPU.

### Alojamiento en la Nube
- **Descripción**: Utilizar un proveedor de la nube para desplegar una instancia que aloje un modelo específico.
- **Ejemplo**: Ejecutar LLM en proveedores de la nube como [AWS](https://aws.amazon.com/es/?nc2=h_lg), [Azure](https://azure.microsoft.com/es-es), [GCP](https://cloud.google.com/?hl=es) y otros.
- **Ventajas**: Óptimo para personalizar modelos y su tiempo de ejecución (ej. ajustar un modelo para tu caso de uso).

### API Alojada
- **Descripción**: Llamar a LLMs directamente a través de una API.
- **Proveedores**: Incluye [AWS Bedrock](https://aws.amazon.com/es/bedrock/), [Replicate](https://replicate.com/), [Anyscale](https://www.anyscale.com/), [Together](https://www.together.ai/), entre otros. Tampien es el esquema utilizado por [OpenAi](https://openai.com/) para sus modelos.
- **Ventajas**: La opción más sencilla en general.
- **Tabla Comparativa**: [Tabla Comparativa de Rendimiento](https://github.com/ray-project/llmperf-leaderboard)

## 2. Información de Modelos Específicos

Este apartado provee enlaces y detalles sobre diversos LLMs para su uso en distintos contextos.

### 2.1 WEB the Bloke

Recopilatorio de modelos para uso en modo local.

- [TheBloke en Hugging Face](https://huggingface.co/TheBloke)

### 2.2 Modelos Mistral7b

Información y documentación sobre Modelos Mistral7b.

- [Documentación de Modelos Mistral7b](https://docs.mistral.ai/models/)

### 2.3 Modelos Llama2

Información y documentación sobre Modelos Llama2.

- [Documentación de Llama 2 en AI Meta](https://ai.meta.com/llama/)


### 2.4 Modelos GPT

Información y documentación sobre Modelos GPT.

- [Documentación de Modelos GPT en OpenAI Platform](https://platform.openai.com/docs/models)

## Instalación

Instrucciones paso a paso sobre cómo instalar y configurar el proyecto.

### DOCKER Modelos LLM 

Uso practico de [`llama.cpp`](https://github.com/ggerganov/llama.cpp)

Imagen Docker: [ghcr.io/abetlen/llama-cpp-python](https://github.com/abetlen/llama-cpp-python/pkgs/container/llama-cpp-python)

Los parámetros de configuración de los modelos son los siguientes:

| Opción | Descripción | Valor por defecto |
| --- | --- | --- |
| -h, --help | Muestra este mensaje de ayuda y sale | N/A |
| --model MODEL | Ruta al modelo a utilizar para generar completaciones | PydanticUndefined |
| --model_alias MODEL_ALIAS | El alias del modelo a utilizar para generar completaciones | N/A |
| --n_gpu_layers N_GPU_LAYERS | El número de capas para colocar en la GPU. El resto estará en la CPU. Establezca -1 para mover todo a la GPU | 0 |
| --main_gpu MAIN_GPU | GPU principal a utilizar | 0 |
| --tensor_split [TENSOR_SPLIT ...] | Divide las capas en múltiples GPUs en proporción | N/A |
| --vocab_only VOCAB_ONLY | Si solo se debe devolver el vocabulario | False |
| --use_mmap USE_MMAP | Usa mmap | True |
| --use_mlock USE_MLOCK | Usa mlock | True |
| --seed SEED | Semilla aleatoria. -1 para aleatorio | 4294967295 |
| --n_ctx N_CTX | Tamaño del contexto | 2048 |
| --n_batch N_BATCH | Tamaño del lote a utilizar por evaluación | 512 |
| --n_threads N_THREADS | Número de hilos a utilizar | 8 |
| --n_threads_batch N_THREADS_BATCH | Número de hilos a utilizar al procesar por lotes | 8 |
| --rope_scaling_type ROPE_SCALING_TYPE | Tipo de escalado de RoPE | N/A |
| --rope_freq_base ROPE_FREQ_BASE | Frecuencia base de RoPE | 0.0 |
| --rope_freq_scale ROPE_FREQ_SCALE | Factor de escalado de frecuencia de RoPE | 0.0 |
| --yarn_ext_factor YARN_EXT_FACTOR | Factor de extensión de YARN | N/A |
| --yarn_attn_factor YARN_ATTN_FACTOR | Factor de atención de YARN | N/A |
| --yarn_beta_fast YARN_BETA_FAST | Beta rápido de YARN | N/A |
| --yarn_beta_slow YARN_BETA_SLOW | Beta lento de YARN | N/A |
| --yarn_orig_ctx YARN_ORIG_CTX | Contexto original de YARN | N/A |
| --mul_mat_q MUL_MAT_Q | Si es cierto, usa núcleos experimentales de mul_mat_q | True |
| --f16_kv F16_KV | Si se debe usar clave/valor f16 | True |
| --logits_all LOGITS_ALL | Si se deben devolver los logits | True |
| --embedding EMBEDDING | Si se deben usar embeddings | True |
| --last_n_tokens_size LAST_N_TOKENS_SIZE | Últimos n tokens a mantener para el cálculo de la penalización por repetición | 64 |
| --lora_base LORA_BASE | Ruta opcional al modelo base, útil si se utiliza un modelo base cuantizado y se desea aplicar LoRA a un modelo f16 | N/A |
| --lora_path LORA_PATH | Ruta a un archivo LoRA para aplicar al modelo | N/A |
| --numa NUMA | Habilitar soporte NUMA | False |
| --chat_format CHAT_FORMAT | Formato de chat a utilizar | llama-2 |
| --clip_model_path CLIP_MODEL_PATH | Ruta a un modelo CLIP para utilizar en la completación de chat multimodal | N/A |
| --cache CACHE | Usar una caché para reducir los tiempos de procesamiento de las solicitudes evaluadas | False |
| --cache_type CACHE_TYPE | El tipo de caché a utilizar. Solo se utiliza si la caché está habilitada | ram |
| --cache_size CACHE_SIZE | El tamaño de la caché en bytes. Solo se utiliza si la caché está habilitada | 2147483648 |
| --verbose VERBOSE | Si imprimir información de depuración | True |
| --host HOST | Dirección de escucha | localhost |
| --port PORT | Puerto de escucha | 8000 |
| --interrupt_requests INTERRUPT_REQUESTS | Si interrumpir las solicitudes cuando se recibe una nueva solicitud | True 

### DOCKER Neo4j

Esquema de servicio Docker utilizado para mantener la estructura de árboles de la App. (Ver nota)

![Esquema Neo4jDocker](/ESQUEMAS_GRAFICOS/NEOEsquema1.png)

### Aplicación WEB

Esquema conceptual aplicativo construido

![Esquema conceptual](/ESQUEMAS_GRAFICOS/APLICATIVO_WEB.png)

Imagen de la aplicación WEB construida

![imagen WEB](/ESQUEMAS_GRAFICOS/WEB.png)

**Nota:** Para el funcionamiento del programa es necesario tener arrancado un servicio DOCKER con Neo4j.
Dentro de esta BBDD estará la estructura de datos de la app.

No se proporciona el DUMP de estos datos por motivos de confidencialidad.|
