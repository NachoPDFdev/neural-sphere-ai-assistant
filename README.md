# Profesor IA - Asistente Virtual Interactivo

Un asistente virtual interactivo con interfaz gráfica 3D que combina reconocimiento de voz, síntesis de voz y procesamiento de lenguaje natural utilizando LM Studio.

## Características Principales

### Interfaz Visual
- Visualización 3D interactiva con una esfera neural
- Animaciones de partículas y anillos de energía
- Red neuronal visual con conexiones dinámicas
- Interfaz adaptativa que responde a la interacción del usuario
- Indicadores visuales de actividad y estado

### Interacción por Voz
- Reconocimiento de voz en español
- Síntesis de voz con velocidad ajustable
- Sistema de perfiles de voz para múltiples usuarios
- Detección y registro automático de usuarios
- Personalización de colores por usuario

### Configuración de Audio
- Selección de dispositivos de entrada/salida de audio
- Control de volumen integrado
- Múltiples velocidades de voz
- Soporte para diferentes voces del sistema

### Procesamiento de Lenguaje
- Integración con LM Studio para procesamiento de lenguaje natural
- Mantenimiento de contexto de conversación
- Respuestas contextuales y personalizadas
- Soporte para comandos de voz predefinidos

### Características Técnicas
- Sistema de logging para diagnóstico
- Manejo de errores robusto
- Optimización de rendimiento
- Sistema de caché para fuentes y superficies
- Multithreading para operaciones asíncronas

## Controles
- **WASD**: Mover la cámara
- **Rueda del ratón**: Zoom
- **Click + Arrastrar**: Rotar vista 3D
- **+/-**: Ajustar volumen
- **ESPACIO**: Activar habla
- **P**: Menú de dispositivos de audio
- **K**: Menú de perfiles de voz
- **ESC**: Salir

## Requisitos del Sistema
- Python 3.x
- Pygame
- PyTTSx3
- SpeechRecognition
- Sounddevice
- NumPy
- PIL
- OpenAI (configurado para LM Studio)

## Configuración
1. Asegúrate de tener LM Studio ejecutándose en `http://localhost:1234`
2. Instala las dependencias: `pip install -r requirements.txt`
3. Ejecuta el programa: `python profesor.py`

## Registro de Usuarios
Para registrar un nuevo usuario, simplemente di "me llamo [nombre]" y el sistema creará automáticamente un perfil personalizado.

## Personalización
- El sistema permite personalizar colores por usuario
- Velocidades de voz ajustables
- Múltiples dispositivos de audio soportados
- Perfiles de voz guardados automáticamente

## Notas
- El sistema requiere conexión a internet para el reconocimiento de voz
- LM Studio debe estar ejecutándose para el procesamiento de lenguaje natural
- Se recomienda usar un micrófono de buena calidad para mejor reconocimiento de voz

## Licencia
[Incluir información de licencia aquí]

## Tutorial de Ejecución en PowerShell

1. Abre PowerShell y navega al directorio del proyecto:

```powershell
cd r
```
