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
| --interrupt_requests INTERRUPT_REQUESTS | Si interrumpir las solicitudes cuando se recibe una nueva solicitud | True |