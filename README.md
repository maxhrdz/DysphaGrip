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

3. Cierra el monitor serie del Arduino y ejecuta `python sensor_fusion_live.py --desktop` para abrir el tablero de escritorio. Si se solicita, elige el número del puerto del Arduino. Mantén los sensores quietos y el entorno en silencio hasta que aparezca `Detection active`. La calibración dura al menos cinco segundos.
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

## Identificar la actividad en el tablero

Con ambos sensores conectados, ejecuta el siguiente comando y pulsa **Record** en el navegador:

```sh
python sensor_fusion_live.py --record-session
```

El panel **Activity estimate** muestra la estimación automática del último segundo de audio y movimiento: **Swallowing**, **Speaking**, **Head movement**, **Coughing** o **Rest**. El modelo agrupa saliva y agua como deglución. **Uncertain** indica que las señales son ambiguas o que la ventana tiene pocas lecturas de movimiento. **Waiting for signal** aparece al arrancar, durante la calibración o al interrumpirse la señal.

El modelo predeterminado `models/activity_voice_cough.json` usa las vibraciones periódicas y los cambios rápidos de energía para distinguir habla y tos. Si la duda incluye otra actividad, mantiene la incertidumbre. Se puede volver al modelo anterior con `--activity-model models/activity_pilot.json`.

Cuando aparecen dos estimaciones seguidas de habla y a continuación una ventana ambigua, el tablero puede mostrar **Speaking (recent)** durante esa actualización, hasta medio segundo después de la última estimación clara. Después de una tos, una ventana ambigua vuelve inmediatamente a **Uncertain**. Una nueva estimación clara aparece de inmediato. En una sesión grabada, `activity_at_marks.csv` guarda tanto la acción marcada como la estimación que mostraba el tablero al pulsar. En `activity_predictions.csv`, las columnas `estimated_activity` y `displayed_activity` distinguen el cálculo de la ventana y el texto visible.

El ajuste de movimiento y tos compara cómo sube el sonido del micrófono con la intensidad y la dirección del giro que registra el MPU6050. Se aplica cuando el modelo principal propone movimiento de cabeza o tos. Puedes comparar con la versión anterior añadiendo `--no-motion-cough` al comando del tablero. En `activity_at_marks.csv`, revisa las columnas `manual_activity` y `displayed_activity` para saber qué viste al pulsar.

El contador **Candidate events** del navegador (**Audio + motion triggers** en el escritorio) muestra cuántas veces ambas señales superaron sus umbrales. Las teclas registran por separado la acción que realmente hiciste. En una sesión grabada, `activity_predictions.csv` guarda las estimaciones en vivo con el instante final de cada ventana. Al cerrar, `activity_report.json` compara la predicción del segundo anterior a cada marca con la acción indicada, e informa por categoría los aciertos, confusiones y resultados inciertos. Para elegir otro modelo de actividades se puede pasar `--activity-model ruta/modelo.json`.

Para evaluar el ajuste específico de habla y tos, el entrenador también acepta `--voice-baseline models/activity_pilot.json`. Esta opción conserva el clasificador original y entrena con ejemplos de habla y tos un segundo clasificador que mide periodicidad y cambios de energía. Indica un modelo de salida distinto del original y pasa las carpetas de las sesiones de entrenamiento originales junto con las nuevas. El informe incluye una comparación con los mismos pesos de partida.

Para entrenar otra versión y medir los aciertos por actividad, pasa las carpetas de al menos dos sesiones terminadas con capturas etiquetadas. Sustituye los nombres de ejemplo y usa un archivo de salida nuevo para conservar el piloto.

```sh
python train_activity_model.py --session experiments/SESION_A --session experiments/SESION_B --model models/actividades_v2.json --report experiments/validacion_actividades_v2.json
```

Para reproducir la revisión de tos y movimiento de las 10:34, el siguiente comando genera un modelo y un informe nuevos en la carpeta local de experimentos. Reserva las marcas finales a partir de la 17 y reproduce las ventanas que se mostraban al pulsar las teclas.

```sh
python train_motion_cough.py --primary models/activity_voice_cough.json --test-session experiments/20261115_103418_7n_erac4 --first-test-mark 17 --model experiments/motion_cough_retrained.json --report experiments/motion_cough_retrained_report.json --replay --development-session
```

El informe de `train_activity_model.py` prueba cada sesión con un modelo entrenado en las demás. También mide por separado el tramo final de la sesión más reciente, después de entrenar con su tramo inicial y las sesiones anteriores. Las cifras se calculan sobre las acciones marcadas. El modelo final utiliza todas las capturas válidas.

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

## Browser monitor

The local web app uses `python web_monitor.py` on port 8770. When the app launcher is running, it manages this process. The service starts idle. **Record** opens the M-Audio input and Arduino serial port and waits for samples from both. Close the desktop or Arduino Serial Monitor before starting. When multiple boards are attached, set `DYSPHAGRIP_ARDUINO_PORT` to the intended serial device before launching. In standby, USB detection refreshes every two seconds. The detected Arduino port indicates that the board is visible. The MPU6050 shows a live signal only after valid six-axis messages arrive. If no port appears, use a USB data cable and retry after the board finishes booting.

Wait for calibration to label an activity. The browser shows the activity model used by the desktop monitor and plots the recent contact audio and motion. **Stop** finalizes the local files in `experiments/` and then publishes `web_recording.json` for the history page. Wait for the stopped state before starting a new session or exiting the app.

### Opening the English dashboard on macOS

For a fresh checkout, install Python 3.13 and Node.js 22 or 24 with npm. Clone the two branches into the same parent folder, then start the dashboard from the sensor project.

```sh
git clone --branch ai https://github.com/maxhrdz/test2.git prueba2
git clone --branch webpage https://github.com/XxPosedd/dysphagrip.git dysphagrip
cd prueba2
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
./"Open DysphaGrip.command"
```

The launcher installs the web dependencies on first use. If the repositories are stored separately, set `DYSPHAGRIP_WEB_ROOT` to the web checkout. Use **Configure providers.command** to enter your Bright Data token and zone, ElevenLabs key and voice ID, and Neon connection string.

Double-click **Open DysphaGrip.command** or run `/opt/anaconda3/bin/python sensor_fusion_live.py --record-session`. The dashboard opens at **http://127.0.0.1:5173**. Click **Record** with the M-Audio interface and Arduino connected. Click **Stop** to finalize the session. Keep the launcher running while using the dashboard. The original Matplotlib monitor opens with `--desktop --record-session`.

The research library retrieves journal metadata through Bright Data Web Unlocker and links to PubMed. New browser recordings save a summary to Neon; audio, movement and labels stay on the Mac. **Configure providers.command** stores provider settings privately. A connected ElevenLabs API key and voice ID enable the two English recording announcements. macOS supplies the backup voice. **Claim Neon database.command** opens the temporary database claim flow. After claiming, enter the new Neon connection string in **Configure providers.command** and restart the launcher.
