# DysphaGrip

DysphaGrip combina el sonido de un micrófono de contacto con el movimiento de un sensor MPU6050 conectado a un Arduino. El detector registra un posible evento de deglución cuando ambas señales superan sus umbrales dentro de una ventana de 0.65 segundos. Es un prototipo experimental.

## Instalar en macOS

Con Python 3.13 instalado, abre Terminal en la carpeta que contiene este README. Crea el entorno e instala las dependencias con estos comandos.

```sh
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Cada vez que abras otra terminal, vuelve a la carpeta del proyecto y ejecuta `source .venv/bin/activate`.

## Comprobar el proyecto sin sensores

```sh
python -m unittest discover -v
```

La suite simula las entradas y escribe los resultados en carpetas temporales. Al finalizar, busca `OK` en la terminal.

## Probar con el micrófono y el Arduino

1. Conecta el micrófono de contacto a la entrada 1 de la interfaz M-Audio AIR 192|6. Conecta la interfaz y el Arduino por USB. Si macOS solicita acceso al micrófono, concédelo a la aplicación desde la que ejecutas Python.
2. El Arduino debe tener cargado un programa que envíe las seis lecturas del MPU6050 a 115200 baudios. Cada muestra ocupa una línea con este formato.

   ```text
   Tilt X: -0.99 Y: 0.00 Z: 0.20 | Turn X: 1.00 Y: 2.00 Z: 3.00
   ```

3. Cierra el monitor serie del Arduino y ejecuta `python sensor_fusion_live.py` para abrir el tablero. Si se solicita, elige el número del puerto del Arduino. Mantén los sensores quietos y el entorno en silencio hasta que aparezca `Detection active`. La calibración dura al menos cinco segundos.
4. Comprueba que el tablero muestre `Connected` para el micrófono y el MPU6050. Verifica por separado que hablar cerca del micrófono o rozarlo suavemente cambia la gráfica de audio, y que mover el MPU6050 cambia las gráficas de movimiento. Cierra la ventana al terminar.

Si prefieres ver los eventos en la terminal, ejecuta `python event_detector_live.py` y detén la sesión con `Ctrl+C`. Ambos modos agregan los eventos al historial `dysphagrip_events.csv` en la carpeta de ejecución.

### Comprobar la frecuencia del Arduino UNO Q

El sketch `firmware/dysphagrip_motion/dysphagrip_motion.ino` conserva las seis lecturas del MPU6050 y solicita una muestra cada 40 ms (25 Hz). Cada línea termina con un número de secuencia y el instante de lectura en milisegundos según la placa.

1. Duplica tu proyecto de Arduino para conservar la versión anterior. En la copia, reemplaza el contenido completo del sketch por el archivo indicado. Compílalo para UNO Q y cárgalo cuando quieras probarlo.
2. Cierra el monitor serie y el tablero. En la terminal, con el entorno del proyecto activo, ejecuta `python test_arduino_connection.py` y selecciona el puerto de la UNO Q si se solicita.
3. Compara la frecuencia que indica la placa con las lecturas que recibe la Mac. Con este sketch se esperan valores cercanos a 25 por segundo y un intervalo de unos 40 ms en la placa. Los saltos en la secuencia indican muestras que no llegaron al diagnóstico. Cuando termines de comprobar la frecuencia, puedes volver a abrir el tablero.

## Registrar un experimento

```sh
python sensor_fusion_live.py --experiment
```

Tras la calibración, haz clic en la ventana del tablero para que reciba el teclado. Justo después de cada acción, pulsa su tecla o haz clic en el botón correspondiente. Comprueba que aparezca la confirmación de la etiqueta y su marca en la gráfica, incluso si el detector no cuenta un evento.

Cada marca guarda el audio del micrófono y las seis lecturas de movimiento del Arduino desde dos segundos antes hasta dos segundos después de pulsar. Para comparar movimiento y saliva, repite el movimiento pequeño que quieres evaluar y márcalo con `4`. Marca con `1` cada vez que tragues saliva. Deja unos cinco segundos entre acciones para poder examinarlas por separado. Después de la última marca, espera al menos dos segundos antes de cerrar la ventana.

| Tecla | Etiqueta |
| --- | --- |
| 1 | Deglución de saliva |
| 2 | Deglución de agua |
| 3 | Hablar |
| 4 | Mover la cabeza |
| 5 | Toser |
| 6 | Reposo |

Cierra la ventana para generar el informe. La terminal indica la carpeta de la sesión dentro de `experiments/`. Abre `summary.csv` para ver los aciertos, las falsas alarmas y las detecciones omitidas. Puedes revisar las etiquetas manuales en `ground_truth.csv`, las detecciones en `predictions.csv` y sus correspondencias en `matches.csv`. La tolerancia temporal predeterminada para comparar etiquetas y detecciones es de 1.5 segundos.

Dentro de esa misma sesión, abre `captures/mark_0001/` para revisar la primera marca. El número de carpeta corresponde al de la etiqueta guardada en el tablero y en `ground_truth.csv`.

| Archivo por marca | Qué contiene |
| --- | --- |
| `audio.wav` | Audio del micrófono de contacto a la frecuencia de muestreo de la sesión. |
| `motion.csv` | Los tres ejes de aceleración y los tres de giro. El tiempo relativo vale cero al pulsar la marca. |
| `metadata.json` | Etiqueta, tiempos y cobertura de los sensores. Permite comprobar si se capturó toda la ventana. |

## Modelo piloto de audio

El tablero puede mostrar dos detectores simultáneos. El contador principal sigue la regla de audio y movimiento. El segundo contador y sus líneas verdes corresponden a un clasificador de audio que busca patrones en ventanas sucesivas de un segundo. Ambos procesan la señal de forma independiente. El modelo corre localmente con NumPy.

Con el Cherub en la entrada 1 de la M-Audio, el Arduino conectado y el entorno de Python activo, abre la primera comparación con este comando desde la carpeta del proyecto.

```sh
python sensor_fusion_live.py --record-session --audio-model models/audio_pilot.json
```

Este modo habilita las etiquetas y guarda el audio continuo del canal de contacto en la carpeta local de la sesión mientras el detector está calibrado y llegan datos de ambos sensores. La opción existente `--experiment` sigue guardando las ventanas de las marcas. Para ver el modelo sin guardar el audio continuo, usa únicamente `--audio-model models/audio_pilot.json`.

1. Espera a que ambos sensores indiquen `Connected` y aparezca `Detection active`. Comprueba la frecuencia del Arduino con el diagnóstico descrito arriba antes de la prueba.
2. Alterna acciones y márcalas inmediatamente después. Para una primera sesión de prueba, procura registrar cinco ejemplos separados de saliva, habla, movimiento de cabeza, tos y reposo. Para reposo, permanece quieto unos segundos y pulsa `6`. Deja al menos cinco segundos entre marcas y dos segundos antes de cerrar. Marca todas las acciones para poder interpretar las falsas alarmas.
3. Cierra el tablero. La terminal imprime los aciertos, las omisiones y las falsas alarmas por minuto de ambos detectores. En la carpeta de la sesión, abre `audio_comparison.png` para ver las marcas manuales y las detecciones, o `audio_comparison.json` para revisar los tiempos y las métricas. El cálculo comprende solo los tramos con señal evaluable para ambos detectores.

El modelo piloto se entrenó con marcas de una sola sesión. Guarda sesiones completas distintas para entrenamiento y evaluación. Para entrenar otra versión usa una ruta nueva de salida y repite `--session` por cada sesión de entrenamiento. Reserva una sesión grabada con `--record-session` para comparar. Los nombres en mayúsculas son ejemplos que debes sustituir.

```sh
python train_audio_model.py train --session experiments/ENTRENAMIENTO --model models/mi_modelo.json
python train_audio_model.py compare --session experiments/PRUEBA --model models/mi_modelo.json
```

Las marcas son referencias manuales. Un candidato indica que la señal se parece a un ejemplo etiquetado. La capacidad de recuperar acciones nuevas se mide en la sesión reservada.
