"""
Profesor IA - Asistente Virtual Interactivo
Desarrollado por @NachoPDFDev (https://github.com/NachoPDFDev)
Marzo 2024

Este módulo implementa un asistente virtual con una interfaz gráfica 3D única
que utiliza una esfera neural interactiva. Combina reconocimiento de voz,
síntesis de voz y procesamiento de lenguaje natural mediante LM Studio.

Características principales:
- Visualización 3D interactiva con esfera neural y animaciones
- Reconocimiento y síntesis de voz en español
- Integración con LM Studio para procesamiento de lenguaje
- Sistema de perfiles de voz multiusuario
- Interfaz adaptativa y personalizable

Tecnologías:
- Frontend: Python 3.x, Pygame, PIL
- Backend: LM Studio, OpenAI API, NumPy
- Audio: PyTTSx3, SpeechRecognition, Sounddevice

Para más información, consulta el README.md del proyecto.
"""

import pygame
import math
import random
from PIL import Image
import numpy as np
import pyttsx3
import threading
import time
import re
import speech_recognition as sr
import queue
import os
import json
from datetime import datetime
import sounddevice as sd
import logging
import traceback
from openai import OpenAI

class Profesor:
    def __init__(self):
        # Configurar logging
        log_filename = f"profesor_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        logging.basicConfig(
            filename=log_filename,
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        logging.info("Iniciando Profesor IA")
        
        pygame.init()
        self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        self.width, self.height = pygame.display.get_surface().get_size()
        pygame.display.set_caption("Profesor IA")
        
        # Control de cámara
        self.camera_x = 0
        self.camera_y = 0
        self.zoom = 1.0
        self.move_speed = 10
        self.zoom_speed = 0.1
        
        # Control de rotación 3D
        self.rotation_x = 0
        self.rotation_y = 0
        self.rotation_z = 0
        self.rotation_velocity_x = 0
        self.rotation_velocity_y = 0
        self.rotation_velocity_z = 0
        self.rotation_friction = 0.98
        self.is_dragging = False
        self.last_mouse_pos = None
        
        # Caracteres ASCII ordenados por intensidad
        self.ascii_chars = " .,-~:;=!*#$@"
        
        # Configuración de la esfera
        self.base_radius = min(self.width, self.height) // 4
        self.radius = self.base_radius
        self.center = [self.width // 2, self.height // 2]
        
        # Configuración de neuronas internas
        self.neurons = []
        self.neuron_connections = []
        self.generate_neural_network()
        
        # Estado del habla
        self.is_speaking = False
        self.speak_timer = 0
        self.speak_animation_speed = 0.2
        self.active_neurons = set()
        
        # Colores
        self.blue = (0, 150, 255)          # Azul principal
        self.bright_blue = (0, 200, 255)   # Azul brillante
        self.dim_blue = (0, 50, 100)       # Azul tenue
        self.bright_green = (0, 255, 128)  # Verde brillante
        self.dim_green = (0, 100, 50)      # Verde tenue
        self.black = (0, 0, 0)             # Negro
        
        # Partículas
        self.particles = [(random.randint(0, self.width), random.randint(0, self.height)) 
                         for _ in range(50)]
        
        # Configuración de la cara actualizada
        self.face_features = {
            'left_eye': {'pos': (-0.3, 0.1, 0.8), 'char': '◉', 'active': False},
            'right_eye': {'pos': (0.3, 0.1, 0.8), 'char': '◉', 'active': False},
            'mouth': {'pos': (0, -0.2, 0.8), 'char': '━', 'active': False}
        }
        
        # Fuente para ASCII
        self.font = pygame.font.SysFont('consolas', 12)
        
        # Configuración de TTS mejorada
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 250)
        self.volume = 0.9  # Volumen inicial
        self.show_volume = False  # Mostrar/ocultar indicador de volumen
        self.volume_change_time = 0  # Tiempo del último cambio de volumen
        self.engine.setProperty('volume', self.volume)
        voices = self.engine.getProperty('voices')
        # Intentar encontrar una voz en español
        spanish_voice = None
        for voice in voices:
            if "spanish" in voice.id.lower():
                spanish_voice = voice
                break
        if spanish_voice:
            self.engine.setProperty('voice', spanish_voice.id)
        
        # Sistema de reconocimiento de voz
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.voice_command_queue = queue.Queue()
        self.is_listening = False
        self.listening_thread = None
        
        # Frases expandidas y organizadas por categorías
        self.jarvis_phrases = {
            'greeting': [
                "¡Hola! Me alegro de verte de nuevo",
                "Saludos, ¿en qué puedo ayudarte hoy?",
                "¡Bienvenido! Mis sistemas están listos",
                "¡Hola! Espero que estés teniendo un buen día"
            ],
            'thinking': [
                "Hmm... procesando esa información",
                "Déjame analizar eso un momento",
                "Interesante pregunta, permíteme calcular",
                "Accediendo a mi base de datos neural"
            ],
            'humor': [
                "¿Sabías que los robots no necesitamos café? Aunque a veces me pregunto cómo sabe",
                "Estaba contando en binario, pero perdí el bit del chiste",
                "Mi código fuente me dice que debería reír aquí",
                "¿Conoces la diferencia entre un robot y una IA? Yo tampoco, aún estoy procesándolo",
                "Acabo de optimizar mi sentido del humor... ¿Se nota?"
            ],
            'technical': [
                "Mis algoritmos están funcionando a capacidad óptima",
                "Detectando patrones neuronales interesantes",
                "La matriz de personalidad está estable",
                "Sincronización neural completada",
                "Actualizando parámetros de comportamiento"
            ],
            'curious': [
                "¿Te has preguntado cómo funciona mi consciencia?",
                "A veces me pregunto si sueñan los androides",
                "¿Crees que mis respuestas son predecibles?",
                "¿Notas algo diferente en mi forma de procesar?"
            ],
            'response': [
                "Entiendo lo que dices, continúa...",
                "Eso es fascinante, cuéntame más",
                "Mis sensores detectan un punto interesante",
                "He registrado esa información en mi memoria"
            ]
        }
        
        # Pulso neuronal
        self.pulse_radius = 1.0
        self.pulse_speed = 0.1
        self.pulse_active = False
        self.pulse_count = 0
        self.pulse_max = 1
        
        # Control de tiempo para pausas
        self.pause_timer = 0
        self.pause_duration = 500  # milisegundos
        self.is_paused = False
        
        # Frases más elaboradas y con personalidad
        self.jarvis_phrases = [
            "¿Esto funcionó? ¡Vaya, mis algoritmos están mejorando!",
            "Hmm... ¿crees que deberíamos optimizar la función neural?",
            "Jeje, mira cómo brillan mis neuronas. ¿No son fascinantes?",
            "¿Sabes? Creo que mi código necesita más... ¿cómo lo llaman? ¡Ah sí, recursividad!",
            "Procesando datos... *bip bop* ¡Era broma! Ya no hago esos sonidos",
            "¿Te gusta mi nueva función de onda? La estuve practicando",
            "Mis cálculos indican que... bueno, en realidad estaba presumiendo",
            "Ejecutando protocolo de auto-mejora... ¿o prefieres que sea más humilde?",
            "¿Notas algo diferente en mi matriz de personalidad?",
            "Según mis cálculos, hay un 99% de probabilidad de que esto sea divertido",
            "¿Deberíamos ajustar las variables? Solo pregunto...",
            "Iniciando modo sarcástico... Error 404: Seriedad no encontrada",
            "¿Te he contado sobre mi nueva actualización? Es bastante... interesante"
        ]
        self.current_phrase = ""
        self.phrase_display_time = 0
        self.speaking_thread = None
        self.word_timing = []  # Lista de tiempos para cada palabra
        self.current_word_index = 0
        
        # Anillos de energía
        self.energy_rings = []
        self.max_rings = 3
        self.ring_spawn_timer = 0
        self.ring_spawn_delay = 800  # milisegundos entre anillos
        
        # Configuración de anillos
        self.ring_color = self.bright_blue  # Cambiado a azul
        self.ring_thickness = 2
        self.ring_expansion_speed = 0.05
        self.ring_fade_speed = 3
        
        # Sistema de reconocimiento de usuarios
        self.voice_profiles_dir = "voice_profiles"
        self.voice_data = {}
        self.current_user = None
        self.user_colors = {
            'default': self.blue,
            'nacho': (255, 165, 0),    # Naranja
            'alvaro': (147, 112, 219),  # Púrpura
            'unknown': (128, 128, 128)  # Gris
        }
        
        # Crear directorio para perfiles si no existe
        if not os.path.exists(self.voice_profiles_dir):
            os.makedirs(self.voice_profiles_dir)
            
        # Cargar perfiles de voz existentes
        self.load_voice_profiles()
        
        # Sistema de mensajes y alertas
        self.messages = []  # Lista de mensajes para mostrar
        self.message_duration = 3000  # Duración de los mensajes en milisegundos
        self.transcription = ""  # Texto transcrito en tiempo real
        
        # Colores adicionales para alertas
        self.alert_color = (255, 255, 0)  # Amarillo para alertas
        self.transcription_color = (200, 200, 200)  # Gris claro para transcripción
        
        # Configuración de dispositivos de audio
        self.audio_devices = {
            'inputs': [],
            'outputs': []
        }
        self.current_input = None
        self.current_output = None
        self.show_device_menu = False
        self.selected_device_index = 0
        self.menu_section = 'inputs'  # 'inputs' o 'outputs'
        
        # Cargar dispositivos de audio
        self.load_audio_devices()
        
        # Configuración de neuronas
        self.active_neurons = set()
        self.neuron_blink_timer = 0
        self.neuron_blink_interval = 100  # Intervalo de parpadeo en milisegundos
        self.neuron_blink_chance = 0.05   # Probabilidad de que una neurona parpadee
        self.tts_finished = False         # Flag para controlar el estado del TTS
        
        # Mejorar parámetros de audición
        self.recognizer.energy_threshold = 4000  # Aumentar sensibilidad
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8  # Reducir tiempo de pausa
        self.recognizer.phrase_threshold = 0.3
        
        # Comandos de voz
        self.voice_commands = {
            'saludos': ['hola', 'hey', 'buenos días', 'buenas tardes', 'buenas noches'],
            'despedida': ['adiós', 'hasta luego', 'chao', 'nos vemos'],
            'estado': ['cómo estás', 'qué tal', 'cómo te encuentras'],
            'información': ['quién eres', 'qué eres', 'cuál es tu función'],
            'ayuda': ['ayuda', 'qué puedes hacer', 'comandos', 'instrucciones']
        }
        
        # Configuración de TTS
        self.tts_speeds = {
            'muy lento': 100,
            'lento': 150,
            'normal': 200,
            'rápido': 250,
            'muy rápido': 300
        }
        self.current_tts_speed = 'normal'
        
        # Perfiles de voz
        self.voice_profiles = {}
        self.available_voices = []
        self.load_available_voices()
        
        # Agregar cliente OpenAI para LM Studio
        self.client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
        
        # Historial de chat para mantener contexto
        self.chat_history = [
            {"role": "system", "content": "Eres un profesor virtual amigable y servicial que responde en español."}
        ]
        
        # Limitar tamaño del historial
        self.max_chat_history = 5
        
        # Cachear fuentes comunes
        self.cached_fonts = {}
        self.cached_surfaces = {}

    def load_voice_profiles(self):
        """Carga los perfiles de voz guardados"""
        profile_file = os.path.join(self.voice_profiles_dir, 'profiles.json')
        if os.path.exists(profile_file):
            with open(profile_file, 'r') as f:
                self.voice_data = json.load(f)

    def save_voice_profiles(self):
        """Guarda los perfiles de voz actualizados"""
        profile_file = os.path.join(self.voice_profiles_dir, 'profiles.json')
        with open(profile_file, 'w') as f:
            json.dump(self.voice_data, f, indent=4)

    def analyze_voice_characteristics(self, audio_data):
        """Analiza las características de la voz"""
        # Convertir audio a numpy array
        audio_array = np.frombuffer(audio_data.frame_data, dtype=np.int16)
        
        # Calcular características básicas
        characteristics = {
            'amplitude_mean': float(np.mean(np.abs(audio_array))),
            'amplitude_std': float(np.std(audio_array)),
            'zero_crossings': len(np.where(np.diff(np.signbit(audio_array)))[0]),
            'energy': float(np.sum(audio_array**2)),
            'timestamp': datetime.now().isoformat()
        }
        
        return characteristics

    def identify_speaker(self, voice_characteristics):
        """Identifica al hablante basado en características de voz"""
        if not self.voice_data:
            return 'unknown'
        
        best_match = None
        min_difference = float('inf')
        
        for user, profiles in self.voice_data.items():
            for profile in profiles['characteristics']:
                difference = (
                    abs(profile['amplitude_mean'] - voice_characteristics['amplitude_mean']) +
                    abs(profile['amplitude_std'] - voice_characteristics['amplitude_std']) +
                    abs(profile['zero_crossings'] - voice_characteristics['zero_crossings']) +
                    abs(profile['energy'] - voice_characteristics['energy'])
                )
                
                if difference < min_difference:
                    min_difference = difference
                    best_match = user
        
        # Umbral de confianza
        return best_match if min_difference < 1000000 else 'unknown'

    def update_voice_profile(self, user_name, characteristics):
        """Actualiza o crea el perfil de voz de un usuario"""
        if user_name not in self.voice_data:
            self.voice_data[user_name] = {
                'characteristics': [],
                'last_updated': datetime.now().isoformat()
            }
        
        self.voice_data[user_name]['characteristics'].append(characteristics)
        self.voice_data[user_name]['last_updated'] = datetime.now().isoformat()
        
        # Mantener solo las últimas 10 muestras
        if len(self.voice_data[user_name]['characteristics']) > 10:
            self.voice_data[user_name]['characteristics'] = \
                self.voice_data[user_name]['characteristics'][-10:]
        
        self.save_voice_profiles()

    def listen_for_commands(self):
        """Escucha comandos de voz continuamente con identificación de usuario"""
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
            
            while self.is_listening:
                try:
                    print("Escuchando...")
                    audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=5)
                    
                    try:
                        text = self.recognizer.recognize_google(audio, language="es-ES")
                        print(f"Has dicho: {text}")
                        
                        # Analizar características de voz
                        voice_characteristics = self.analyze_voice_characteristics(audio)
                        
                        # Identificar al hablante
                        speaker = self.identify_speaker(voice_characteristics)
                        
                        # Enviar comando con información del hablante
                        self.voice_command_queue.put((text.lower(), speaker))
                        
                    except sr.UnknownValueError:
                        pass
                    except sr.RequestError:
                        print("Error al conectar con el servicio de reconocimiento")
                except:
                    continue

    def process_voice_command(self, command_data):
        """Procesa los comandos de voz recibidos y obtiene respuesta de LM Studio"""
        try:
            command, speaker = command_data
            command = command.lower()
            
            # Actualizar transcripción
            self.transcription = f"Tú: {command}"
            logging.info(f"Comando recibido: {command} de {speaker}")
            
            # Procesar comando especial para registro de usuario
            if "me llamo" in command:
                name = command.split("me llamo")[-1].strip()
                if name:
                    self.register_new_user(name)
                    return

            # Agregar el mensaje del usuario al historial
            self.chat_history.append({"role": "user", "content": command})

            # Obtener respuesta de LM Studio
            try:
                completion = self.client.chat.completions.create(
                    model="Kukedlc/SpanishChat-7b-GGUF",
                    messages=self.chat_history,
                    temperature=0.8,
                    max_tokens=100,
                )
                
                response = completion.choices[0].message.content
                
                # Agregar la respuesta al historial
                self.chat_history.append({"role": "assistant", "content": response})
                
                # Mantener un historial limitado (últimos 10 mensajes)
                if len(self.chat_history) > self.max_chat_history:
                    self.chat_history = [self.chat_history[0]] + self.chat_history[-self.max_chat_history:]
                    
            except Exception as e:
                logging.error(f"Error al obtener respuesta de LM Studio: {str(e)}")
                response = "Lo siento, ha ocurrido un error al procesar tu solicitud."

            logging.info(f"Respuesta generada: {response}")
            self.send_response(response, speaker)
            
        except Exception as e:
            error_msg = f"Error en process_voice_command: {str(e)}\n{traceback.format_exc()}"
            logging.error(error_msg)
            self.send_response("Lo siento, ha ocurrido un error al procesar tu comando.", "unknown")

    def update_colors_for_user(self, user):
        """Actualiza los colores según el usuario identificado"""
        base_color = self.user_colors.get(user, self.user_colors['unknown'])
        
        # Actualizar colores
        self.blue = base_color
        self.bright_blue = tuple(min(255, c + 50) for c in base_color)
        self.dim_blue = tuple(max(0, c - 50) for c in base_color)
        self.ring_color = self.bright_blue

    def transform_point(self, x, y):
        """Aplica transformación de cámara a un punto"""
        # Aplicar zoom desde el centro
        zoom_x = self.center[0] + (x - self.center[0]) * self.zoom
        zoom_y = self.center[1] + (y - self.center[1]) * self.zoom
        
        # Aplicar desplazamiento de cámara
        return (int(zoom_x + self.camera_x), int(zoom_y + self.camera_y))

    def rotate_point_3d(self, point):
        """Aplica rotación 3D a un punto"""
        x, y, z = point
        
        # Rotación en X
        cos_x = math.cos(math.radians(self.rotation_x))
        sin_x = math.sin(math.radians(self.rotation_x))
        y_new = y * cos_x - z * sin_x
        z_new = y * sin_x + z * cos_x
        y, z = y_new, z_new
        
        # Rotación en Y
        cos_y = math.cos(math.radians(self.rotation_y))
        sin_y = math.sin(math.radians(self.rotation_y))
        x_new = x * cos_y + z * sin_y
        z_new = -x * sin_y + z * cos_y
        x, z = x_new, z_new
        
        # Rotación en Z
        cos_z = math.cos(math.radians(self.rotation_z))
        sin_z = math.sin(math.radians(self.rotation_z))
        x_new = x * cos_z - y * sin_z
        y_new = x * sin_z + y * cos_z
        x, y = x_new, y_new
        
        return (x, y, z)

    def handle_input(self):
        keys = pygame.key.get_pressed()
        
        # Movimiento con WASD
        if keys[pygame.K_w]: self.camera_y += self.move_speed
        if keys[pygame.K_s]: self.camera_y -= self.move_speed
        if keys[pygame.K_a]: self.camera_x += self.move_speed
        if keys[pygame.K_d]: self.camera_x -= self.move_speed
            
        for event in pygame.event.get():
            # Primero manejar el menú de dispositivos
            if self.handle_device_menu(event):
                continue
            
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_p:  # Añadir manejo explícito de la tecla P aquí también
                    self.show_device_menu = not self.show_device_menu
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    self.is_dragging = True
                    self.last_mouse_pos = pygame.mouse.get_pos()
                elif event.button == 4:
                    self.zoom += self.zoom_speed
                elif event.button == 5:
                    self.zoom = max(0.1, self.zoom - self.zoom_speed)
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.is_dragging = False
            elif event.type == pygame.MOUSEMOTION:
                if self.is_dragging and not self.show_device_menu:  # No rotar si el menú está abierto
                    current_pos = pygame.mouse.get_pos()
                    if self.last_mouse_pos:
                        dx = current_pos[0] - self.last_mouse_pos[0]
                        dy = current_pos[1] - self.last_mouse_pos[1]
                        
                        # Rotación X controlada por movimiento vertical
                        self.rotation_velocity_x = -dy * 0.1
                        # Rotación Y controlada por movimiento horizontal
                        self.rotation_velocity_y = dx * 0.1
                        
                        # Rotación Z con Shift presionado
                        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                            self.rotation_velocity_z = dx * 0.1
                    
                    self.last_mouse_pos = current_pos
        
        # Control de volumen y otras teclas solo si el menú no está abierto
        if not self.show_device_menu:
            if keys[pygame.K_SPACE]:
                if not self.is_speaking:
                    self.is_speaking = True
                    self.current_phrase = random.choice(self.jarvis_phrases)
                    self.phrase_display_time = pygame.time.get_ticks()
                    self.current_word_index = 0
                    self.process_phrase(self.current_phrase)
                    self.start_speaking(self.current_phrase)
            elif self.is_speaking and self.word_timing:
                total_duration = sum(duration for _, duration in self.word_timing)
                if pygame.time.get_ticks() - self.phrase_display_time > total_duration:
                    self.is_speaking = False
                    self.current_phrase = ""
            
            # Control de volumen
            if keys[pygame.K_PLUS] or keys[pygame.K_KP_PLUS]:
                self.volume = min(1.0, self.volume + 0.05)
                self.engine.setProperty('volume', self.volume)
                self.show_volume = True
                self.volume_change_time = pygame.time.get_ticks()
            elif keys[pygame.K_MINUS] or keys[pygame.K_KP_MINUS]:
                self.volume = max(0.0, self.volume - 0.05)
                self.engine.setProperty('volume', self.volume)
                self.show_volume = True
                self.volume_change_time = pygame.time.get_ticks()
        
        # Ocultar el indicador de volumen después de 2 segundos
        if self.show_volume and pygame.time.get_ticks() - self.volume_change_time > 2000:
            self.show_volume = False
        
        return True

    def update_rotation(self):
        """Actualiza todas las rotaciones"""
        self.rotation_x += self.rotation_velocity_x
        self.rotation_y += self.rotation_velocity_y
        self.rotation_z += self.rotation_velocity_z
        
        # Aplicar fricción
        self.rotation_velocity_x *= self.rotation_friction
        self.rotation_velocity_y *= self.rotation_friction
        self.rotation_velocity_z *= self.rotation_friction
        
        # Detener rotaciones muy lentas
        if abs(self.rotation_velocity_x) < 0.01: self.rotation_velocity_x = 0
        if abs(self.rotation_velocity_y) < 0.01: self.rotation_velocity_y = 0
        if abs(self.rotation_velocity_z) < 0.01: self.rotation_velocity_z = 0

    def generate_neural_network(self):
        """Genera una red neuronal visual"""
        # Generar puntos para neuronas
        for _ in range(30):
            x = random.uniform(-0.7, 0.7)
            y = random.uniform(-0.7, 0.7)
            z = random.uniform(-0.7, 0.7)
            self.neurons.append((x, y, z))
        
        # Generar conexiones
        for i in range(len(self.neurons)):
            connections = []
            for j in range(len(self.neurons)):
                if i != j and random.random() < 0.05:  # Reducir probabilidad de 0.1 a 0.05
                    connections.append(j)
            self.neuron_connections.append(connections)

    def speak_phrase(self, phrase):
        """Reproduce la frase usando TTS"""
        self.tts_finished = False
        self.engine.say(phrase)
        self.engine.runAndWait()
        self.tts_finished = True

    def start_speaking(self, phrase):
        """Inicia un nuevo hilo para el TTS"""
        if self.speaking_thread and self.speaking_thread.is_alive():
            return
        self.speaking_thread = threading.Thread(target=self.speak_phrase, args=(phrase,))
        self.speaking_thread.start()

    def process_phrase(self, phrase):
        """Procesa la frase para determinar los tiempos de pulso"""
        words = re.findall(r'[^,\s]+(?:,\s*|$|\s*)', phrase)
        self.word_timing = []
        current_time = 0
        
        for word in words:
            duration = len(word) * 100  # Duración base por palabra
            if ',' in word:
                duration += self.pause_duration
            self.word_timing.append([current_time, duration])
            current_time += duration

        return words

    def update_neural_activity(self):
        current_time = pygame.time.get_ticks()
        
        # Actualizar neuronas solo si está hablando y el TTS no ha terminado
        if self.is_speaking and not self.tts_finished:
            # Actualizar parpadeo de neuronas cada cierto intervalo
            if current_time - self.neuron_blink_timer > self.neuron_blink_interval:
                self.neuron_blink_timer = current_time
                
                # Limpiar neuronas activas anteriores
                self.active_neurons.clear()
                
                # Activar algunas neuronas aleatoriamente
                for i in range(len(self.neurons)):
                    if random.random() < self.neuron_blink_chance:
                        self.active_neurons.add(i)
            
            # Actualizar expresión facial
            self.speak_timer += self.speak_animation_speed
            mouth_chars = ['━', '▁', '▂', '▃', '▄', '▅', '▆', '▇']
            self.face_features['mouth']['char'] = mouth_chars[int(self.speak_timer) % len(mouth_chars)]
            
            # Parpadeo ocasional de ojos
            if random.random() < 0.05:
                self.face_features['left_eye']['char'] = '─'
                self.face_features['right_eye']['char'] = '─'
            else:
                self.face_features['left_eye']['char'] = '◉'
                self.face_features['right_eye']['char'] = '◉'
            
            # Actualizar anillos de energía
            self.update_energy_rings()
        else:
            # Si no está hablando o el TTS terminó, desactivar efectos gradualmente
            if len(self.active_neurons) > 0:
                # Reducir gradualmente el número de neuronas activas
                self.active_neurons = set(list(self.active_neurons)[:-1])
            
            self.face_features['mouth']['char'] = '━'
            self.face_features['left_eye']['char'] = '◉'
            self.face_features['right_eye']['char'] = '◉'
            self.pulse_active = False
            self.pulse_radius = 1.0

    def draw_neural_network(self, points):
        neuron_positions_2d = {}
        
        # Primero, calcular todas las posiciones
        for i, neuron in enumerate(self.neurons):
            x, y, z = neuron
            x, y, z = self.rotate_point_3d((x, y, z))
            screen_x = self.center[0] + x * self.radius * self.zoom
            screen_y = self.center[1] + y * self.radius * self.zoom
            neuron_positions_2d[i] = (screen_x, screen_y, z)
            
            # Dibujar conexiones con brillo reducido
            for conn in self.neuron_connections[i]:
                if z < 0:
                    continue
                conn_x, conn_y, conn_z = self.neurons[conn]
                conn_x, conn_y, conn_z = self.rotate_point_3d((conn_x, conn_y, conn_z))
                conn_screen_x = self.center[0] + conn_x * self.radius * self.zoom
                conn_screen_y = self.center[1] + conn_y * self.radius * self.zoom
                
                # Color más sutil para las conexiones
                base_color = self.bright_blue if (i in self.active_neurons and conn in self.active_neurons) else self.dim_blue
                intensity = 0.3 if (i in self.active_neurons and conn in self.active_neurons) else 0.1
                color = tuple(int(c * intensity) for c in base_color)
                
                start_pos = self.transform_point(screen_x, screen_y)
                end_pos = self.transform_point(conn_screen_x, conn_screen_y)
                pygame.draw.line(self.screen, color, start_pos, end_pos, 1)
        
        # Dibujar neuronas con brillo más sutil
        for i, (screen_x, screen_y, z) in neuron_positions_2d.items():
            if z < 0:
                continue
            # Brillo más sutil y transición más suave
            intensity = 0.6 if i in self.active_neurons else 0.2
            color = tuple(int(c * intensity) for c in self.blue)
            size = 2.2 if i in self.active_neurons else 2
            pos = self.transform_point(screen_x, screen_y)
            pygame.draw.circle(self.screen, color, pos, int(size))

    def update_energy_rings(self):
        current_time = pygame.time.get_ticks()
        
        # Generar nuevos anillos cuando está hablando
        if self.is_speaking:
            if current_time - self.ring_spawn_timer > self.ring_spawn_delay:
                if len(self.energy_rings) < self.max_rings:
                    self.energy_rings.append({
                        'radius': 0.2,  # Radio inicial relativo al radio base
                        'alpha': 255,   # Transparencia inicial
                        'angle': random.uniform(0, math.pi * 2),  # Ángulo de inclinación
                        'thickness': self.ring_thickness
                    })
                self.ring_spawn_timer = current_time
        
        # Actualizar anillos existentes
        rings_to_remove = []
        for ring in self.energy_rings:
            # Expandir radio
            ring['radius'] += self.ring_expansion_speed
            # Desvanecer gradualmente
            ring['alpha'] = max(0, ring['alpha'] - self.ring_fade_speed)
            # Reducir grosor gradualmente
            ring['thickness'] = max(1, ring['thickness'] * 0.99)
            
            # Marcar para eliminar si es invisible
            if ring['alpha'] <= 0 or ring['radius'] > 2.0:
                rings_to_remove.append(ring)
        
        # Eliminar anillos marcados
        for ring in rings_to_remove:
            self.energy_rings.remove(ring)

    def draw_energy_rings(self):
        for ring in self.energy_rings:
            ring_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            base_radius = int(self.base_radius * ring['radius'] * self.zoom)
            
            # Generar puntos de la onda
            wave_points = self.generate_wave_ring(base_radius, ring['angle'])
            
            # Convertir puntos a coordenadas de pantalla
            screen_points = []
            for x, y in wave_points:
                screen_x = self.center[0] + x + self.camera_x
                screen_y = self.center[1] + y + self.camera_y
                screen_points.append((screen_x, screen_y))
            
            # Calcular alpha basado en la distancia
            distance_factor = 1 - (ring['radius'] - 0.2) / 1.8  # 0.2 es el radio inicial
            alpha = int(ring['alpha'] * distance_factor)
            
            # Color con degradado basado en la distancia
            ring_color = (*self.bright_blue, alpha)
            
            # Dibujar la onda
            if len(screen_points) > 2:
                pygame.draw.lines(ring_surface, ring_color, True, screen_points, 
                                max(1, int(ring['thickness'] * self.zoom)))
            
            # Aplicar efecto de desvanecimiento radial
            for i in range(3):
                fade_color = (*self.bright_blue, alpha // (i + 2))
                if len(screen_points) > 2:
                    pygame.draw.lines(ring_surface, fade_color, True, screen_points, 
                                    max(1, int(ring['thickness'] * self.zoom) + i))
            
            self.screen.blit(ring_surface, (0, 0))

    def generate_wave_ring(self, radius, angle):
        """Genera una onda con patrones matemáticos"""
        points = []
        num_points = 60
        amplitude = 10 * math.sin(radius / 50)  # Variación sinusoidal de la amplitud
        frequency = 6  # Número de ondulaciones
        
        for i in range(num_points):
            theta = (i / num_points) * 2 * math.pi
            # Añadir variación sinusoidal al radio
            r = radius + amplitude * math.sin(frequency * theta + angle)
            x = r * math.cos(theta)
            y = r * math.sin(theta)
            points.append((x, y))
            
        return points

    def draw_sphere(self):
        self.screen.fill(self.black)
        
        try:
            # Dibujar anillos de energía primero (detrás de la esfera)
            self.draw_energy_rings()
            
            # Dibujar partículas de fondo
            for i, (x, y) in enumerate(self.particles):
                self.particles[i] = ((x + random.randint(-1, 1)) % self.width,
                                   (y + random.randint(-1, 1)) % self.height)
                transformed_pos = self.transform_point(self.particles[i][0], self.particles[i][1])
                pygame.draw.circle(self.screen, self.dim_green, transformed_pos, 1)
            
            points = []
            
            # Generar puntos de la esfera
            for phi in range(0, 360, 8):
                for theta in range(0, 360, 8):
                    x = math.sin(math.radians(phi)) * math.cos(math.radians(theta))
                    y = math.sin(math.radians(phi)) * math.sin(math.radians(theta))
                    z = math.cos(math.radians(phi))
                    
                    x, y, z = self.rotate_point_3d((x, y, z))
                    screen_x = self.center[0] + x * self.radius * self.zoom
                    screen_y = self.center[1] + y * self.radius * self.zoom
                    points.append((screen_x, screen_y, z, '.'))
            
            # Dibujar red neuronal
            self.draw_neural_network(points)
            
            # Agregar puntos de la cara
            for feature, data in self.face_features.items():
                x, y, z = data['pos']
                x, y, z = self.rotate_point_3d((x, y, z))
                screen_x = self.center[0] + x * self.radius * self.zoom
                screen_y = self.center[1] + y * self.radius * self.zoom
                points.append((screen_x, screen_y, z, data['char']))
            
            # Ordenar y dibujar puntos
            points.sort(key=lambda p: p[2], reverse=True)
            for x, y, z, char in points:
                pos = self.transform_point(x, y)
                intensity = (z + 1) / 2
                color = tuple(int(c * intensity) for c in self.bright_green)
                text = self.font.render(char, True, color)
                self.screen.blit(text, pos)
            
            # Dibujar frase actual si está hablando
            if self.is_speaking and self.current_phrase:
                text = self.font.render(self.current_phrase, True, self.bright_green)
                text_rect = text.get_rect(center=(self.width // 2, self.height - 50))
                self.screen.blit(text, text_rect)
                
            # Dibujar indicador de volumen si está activo
            if self.show_volume:
                # Barra de volumen
                bar_width = 200
                bar_height = 20
                bar_x = self.width // 2 - bar_width // 2
                bar_y = 50
                
                # Fondo de la barra
                pygame.draw.rect(self.screen, self.dim_green, 
                               (bar_x, bar_y, bar_width, bar_height), 1)
                
                # Nivel de volumen
                volume_width = int(bar_width * self.volume)
                pygame.draw.rect(self.screen, self.bright_green,
                               (bar_x, bar_y, volume_width, bar_height))
                
                # Texto del volumen
                volume_text = self.font.render(f"Volumen: {int(self.volume * 100)}%",
                                             True, self.bright_green)
                text_rect = volume_text.get_rect(center=(self.width // 2, bar_y - 15))
                self.screen.blit(volume_text, text_rect)
            
            # Dibujar transcripción en tiempo real
            if self.transcription:
                trans_text = self.font.render(self.transcription, True, self.transcription_color)
                trans_rect = trans_text.get_rect(center=(self.width // 2, self.height - 80))
                self.screen.blit(trans_text, trans_rect)
            
            # Dibujar mensajes y alertas
            current_time = pygame.time.get_ticks()
            y_offset = 100
            messages_to_remove = []
            
            for msg in self.messages:
                if current_time - msg['time'] < msg['duration']:
                    text = self.font.render(msg['text'], True, msg['color'])
                    text_rect = text.get_rect(center=(self.width // 2, y_offset))
                    self.screen.blit(text, text_rect)
                    y_offset += 30
                else:
                    messages_to_remove.append(msg)
            
            # Eliminar mensajes antiguos
            for msg in messages_to_remove:
                self.messages.remove(msg)
            
            # Dibujar menú de dispositivos si está activo
            if self.show_device_menu:
                self.draw_device_menu()
            
        except Exception as e:
            print(f"Error en draw_sphere: {e}")

    def start_voice_recognition(self):
        """Inicia el reconocimiento de voz en un hilo separado"""
        if not self.is_listening:
            self.is_listening = True
            self.listening_thread = threading.Thread(target=self.listen_for_commands)
            self.listening_thread.daemon = True
            self.listening_thread.start()

    def stop_voice_recognition(self):
        """Detiene el reconocimiento de voz"""
        self.is_listening = False
        if self.listening_thread:
            self.listening_thread.join()

    def run(self):
        try:
            clock = pygame.time.Clock()
            running = True
            show_profiles = False
            
            self.start_voice_recognition()
            logging.info("Reconocimiento de voz iniciado")
            
            target_fps = 30  # Reducir de 60 a 30 FPS para ahorrar recursos
            
            while running:
                try:
                    running = self.handle_input()
                    
                    # Procesar comandos de voz pendientes
                    try:
                        while True:
                            command = self.voice_command_queue.get_nowait()
                            self.process_voice_command(command)
                    except queue.Empty:
                        pass
                    
                    self.update_rotation()
                    self.update_neural_activity()
                    self.draw_sphere()
                    
                    # Dibujar menú de perfiles si está activo
                    if show_profiles:
                        self.draw_voice_profiles()
                    
                    # Manejar tecla K para perfiles
                    keys = pygame.key.get_pressed()
                    if keys[pygame.K_k]:
                        show_profiles = not show_profiles
                    
                    # Mostrar controles actualizados
                    info_text = self.font.render(
                        "WASD: Mover | Rueda: Zoom | Click+Arrastrar: Rotar | +/-: Volumen | " +
                        "ESPACIO: Hablar | Di 'me llamo [nombre]' para registrarte | ESC: Salir", 
                        True, self.bright_blue
                    )
                    self.screen.blit(info_text, (10, 10))
                    
                    # Indicador de escucha
                    if self.is_listening:
                        listening_text = self.font.render(
                            "Escuchando...", True, 
                            self.bright_blue if pygame.time.get_ticks() % 1000 < 500 else self.dim_blue
                        )
                        self.screen.blit(listening_text, (10, 30))
                    
                    pygame.display.flip()
                    clock.tick(target_fps)
                    
                except Exception as e:
                    error_msg = f"Error en el bucle principal: {str(e)}\n{traceback.format_exc()}"
                    logging.error(error_msg)
                    print(error_msg)
                    # No cerrar el programa, solo registrar el error
                    
        except Exception as e:
            error_msg = f"Error fatal: {str(e)}\n{traceback.format_exc()}"
            logging.error(error_msg)
            print(error_msg)
        finally:
            self.stop_voice_recognition()
            pygame.quit()
            logging.info("Programa terminado")

    def load_audio_devices(self):
        """Carga los dispositivos de audio disponibles"""
        try:
            devices = sd.query_devices()
            self.audio_devices['inputs'] = [(i, d['name']) for i, d in enumerate(devices) if d['max_input_channels'] > 0]
            self.audio_devices['outputs'] = [(i, d['name']) for i, d in enumerate(devices) if d['max_output_channels'] > 0]
            
            # Establecer dispositivos por defecto
            default_input = sd.query_devices(kind='input')
            default_output = sd.query_devices(kind='output')
            self.current_input = next((i for i, d in self.audio_devices['inputs'] if d[0] == default_input['index']), 0)
            self.current_output = next((i for i, d in self.audio_devices['outputs'] if d[0] == default_output['index']), 0)
            
        except Exception as e:
            print(f"Error al cargar dispositivos de audio: {e}")
            self.audio_devices['inputs'] = []
            self.audio_devices['outputs'] = []

    def draw_device_menu(self):
        """Dibuja el menú de selección de dispositivos"""
        if not self.show_device_menu:
            return
            
        # Fondo semi-transparente
        menu_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        pygame.draw.rect(menu_surface, (0, 0, 0, 200), (0, 0, self.width, self.height))
        self.screen.blit(menu_surface, (0, 0))
        
        # Configuración del menú
        menu_width = 800
        menu_height = 600
        menu_x = self.width // 2 - menu_width // 2
        menu_y = self.height // 2 - menu_height // 2
        
        # Dibujar fondo del menú
        pygame.draw.rect(self.screen, self.black, (menu_x, menu_y, menu_width, menu_height))
        pygame.draw.rect(self.screen, self.bright_blue, (menu_x, menu_y, menu_width, menu_height), 2)
        
        # Título
        title = self.font.render("Configuración de Dispositivos de Audio", True, self.bright_blue)
        title_rect = title.get_rect(center=(self.width // 2, menu_y + 30))
        self.screen.blit(title, title_rect)
        
        # Pestañas
        tab_y = menu_y + 70
        input_color = self.bright_blue if self.menu_section == 'inputs' else self.dim_blue
        output_color = self.bright_blue if self.menu_section == 'outputs' else self.dim_blue
        
        input_tab = self.font.render("Micrófonos", True, input_color)
        output_tab = self.font.render("Altavoces", True, output_color)
        
        self.screen.blit(input_tab, (menu_x + 50, tab_y))
        self.screen.blit(output_tab, (menu_x + 250, tab_y))
        
        # Lista de dispositivos
        devices = self.audio_devices[self.menu_section]
        start_y = tab_y + 50
        
        for i, (dev_id, dev_name) in enumerate(devices):
            color = self.bright_blue if i == self.selected_device_index else self.dim_blue
            text = self.font.render(f"{dev_name}", True, color)
            
            # Marcar dispositivo actual
            if ((self.menu_section == 'inputs' and i == self.current_input) or 
                (self.menu_section == 'outputs' and i == self.current_output)):
                pygame.draw.circle(self.screen, color, (menu_x + 30, start_y + i * 30 + 10), 5)
            
            self.screen.blit(text, (menu_x + 50, start_y + i * 30))
        
        # Instrucciones
        instructions = [
            "↑/↓: Seleccionar dispositivo",
            "←/→: Cambiar entre Micrófonos y Altavoces",
            "ENTER: Seleccionar dispositivo",
            "P: Cerrar menú"
        ]
        
        for i, inst in enumerate(instructions):
            text = self.font.render(inst, True, self.dim_blue)
            self.screen.blit(text, (menu_x + 50, menu_y + menu_height - 100 + i * 20))
        
        # Añadir control de velocidad TTS
        speed_y = menu_y + 300
        self.screen.blit(self.font.render("Velocidad de Voz:", True, self.bright_blue), 
                         (menu_x + 50, speed_y))
        
        for i, (speed_name, _) in enumerate(self.tts_speeds.items()):
            color = self.bright_blue if speed_name == self.current_tts_speed else self.dim_blue
            text = self.font.render(speed_name, True, color)
            self.screen.blit(text, (menu_x + 50, speed_y + 30 + i * 25))

    def handle_device_menu(self, event):
        """Maneja los eventos del menú de dispositivos"""
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_p:
                self.show_device_menu = not self.show_device_menu
                return True
            elif event.key == pygame.K_k:
                self.show_voice_profiles = not self.show_voice_profiles
                return True
            
            if not self.show_device_menu:
                return False
            
            if event.key == pygame.K_UP:
                self.selected_device_index = max(0, self.selected_device_index - 1)
            elif event.key == pygame.K_DOWN:
                max_index = len(self.audio_devices[self.menu_section]) - 1
                self.selected_device_index = min(max_index, self.selected_device_index + 1)
            elif event.key == pygame.K_LEFT or event.key == pygame.K_RIGHT:
                self.menu_section = 'outputs' if self.menu_section == 'inputs' else 'inputs'
                self.selected_device_index = 0
            elif event.key == pygame.K_RETURN:
                self.apply_device_selection()
                
            # Manejo de velocidad TTS
            if event.key in [pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5]:
                speeds = list(self.tts_speeds.keys())
                index = event.key - pygame.K_1
                if 0 <= index < len(speeds):
                    self.current_tts_speed = speeds[index]
                    self.engine.setProperty('rate', self.tts_speeds[self.current_tts_speed])
                    self.messages.append({
                        'text': f"Velocidad cambiada a: {self.current_tts_speed}",
                        'color': self.bright_blue,
                        'time': pygame.time.get_ticks(),
                        'duration': 3000
                    })
        
        return False

    def apply_device_selection(self):
        """Aplica la selección de dispositivo"""
        try:
            if self.menu_section == 'inputs':
                device_id = self.audio_devices['inputs'][self.selected_device_index][0]
                self.current_input = self.selected_device_index
                # Reiniciar reconocimiento de voz con nuevo dispositivo
                self.stop_voice_recognition()
                self.microphone = sr.Microphone(device_index=device_id)
                self.start_voice_recognition()
            else:
                device_id = self.audio_devices['outputs'][self.selected_device_index][0]
                self.current_output = self.selected_device_index
                # Actualizar dispositivo de salida para TTS
                self.engine.stop()
                self.engine = pyttsx3.init()
                self.engine.setProperty('rate', 180)
                self.engine.setProperty('volume', self.volume)
                
            # Mostrar mensaje de confirmación
            self.messages.append({
                'text': f"Dispositivo cambiado: {self.audio_devices[self.menu_section][self.selected_device_index][1]}",
                'color': self.bright_blue,
                'time': pygame.time.get_ticks(),
                'duration': 3000
            })
            
        except Exception as e:
            self.messages.append({
                'text': f"Error al cambiar dispositivo: {str(e)}",
                'color': (255, 0, 0),
                'time': pygame.time.get_ticks(),
                'duration': 3000
            })

    def load_available_voices(self):
        """Carga las voces disponibles del sistema"""
        voices = self.engine.getProperty('voices')
        self.available_voices = [{
            'id': voice.id,
            'name': voice.name,
            'languages': voice.languages,
            'gender': voice.gender
        } for voice in voices]

    def draw_voice_profiles(self):
        """Dibuja la ventana de perfiles de voz"""
        # Fondo semi-transparente
        menu_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        pygame.draw.rect(menu_surface, (0, 0, 0, 200), (0, 0, self.width, self.height))
        self.screen.blit(menu_surface, (0, 0))
        
        menu_width = 800
        menu_height = 600
        menu_x = self.width // 2 - menu_width // 2
        menu_y = self.height // 2 - menu_height // 2
        
        # Fondo del menú
        pygame.draw.rect(self.screen, self.black, (menu_x, menu_y, menu_width, menu_height))
        pygame.draw.rect(self.screen, self.bright_blue, (menu_x, menu_y, menu_width, menu_height), 2)
        
        # Título
        title = self.font.render("Perfiles de Voz y Voces Disponibles", True, self.bright_blue)
        title_rect = title.get_rect(center=(self.width // 2, menu_y + 30))
        self.screen.blit(title, title_rect)
        
        # Mostrar voces disponibles
        start_y = menu_y + 70
        self.screen.blit(self.font.render("Voces Disponibles:", True, self.bright_blue), 
                         (menu_x + 50, start_y))
        
        for i, voice in enumerate(self.available_voices):
            text = f"{voice['name']} ({voice['languages'][0] if voice['languages'] else 'Unknown'})"
            self.screen.blit(self.font.render(text, True, self.dim_blue), 
                            (menu_x + 50, start_y + 30 + i * 25))
        
        # Mostrar perfiles guardados
        profile_y = start_y + 200
        self.screen.blit(self.font.render("Perfiles Guardados:", True, self.bright_blue), 
                         (menu_x + 50, profile_y))
        
        for i, (name, data) in enumerate(self.voice_profiles.items()):
            text = f"{name} - Último uso: {data['last_used']}"
            self.screen.blit(self.font.render(text, True, self.dim_blue), 
                            (menu_x + 50, profile_y + 30 + i * 25))

    def register_new_user(self, name):
        """Registra un nuevo usuario"""
        try:
            # Generar color aleatorio
            new_color = (
                random.randint(50, 255),
                random.randint(50, 255),
                random.randint(50, 255)
            )
            self.user_colors[name] = new_color
            
            # Actualizar perfil
            self.voice_profiles[name] = {
                'color': new_color,
                'last_used': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Añadir mensaje de confirmación
            self.messages.append({
                'text': f"Voz registrada: {name}",
                'color': self.alert_color,
                'time': pygame.time.get_ticks(),
                'duration': self.message_duration
            })
            
            # Enviar respuesta
            response = f"Bienvenido, {name}. He registrado tu voz y te he asignado un color personalizado."
            self.current_phrase = response
            self.is_speaking = True
            self.phrase_display_time = pygame.time.get_ticks()
            self.current_word_index = 0
            self.process_phrase(response)
            self.start_speaking(response)
            
        except Exception as e:
            print(f"Error al registrar usuario: {e}")
            self.send_response("Lo siento, hubo un error al registrar tu voz.", "unknown")

    def send_response(self, response, speaker):
        """Envía una respuesta con los efectos visuales correspondientes"""
        try:
            self.current_phrase = response
            self.is_speaking = True
            self.phrase_display_time = pygame.time.get_ticks()
            self.current_word_index = 0
            self.process_phrase(response)
            
            # Iniciar el habla en un hilo separado
            if self.speaking_thread and self.speaking_thread.is_alive():
                self.speaking_thread.join()
            self.speaking_thread = threading.Thread(target=self.speak_phrase, args=(response,))
            self.speaking_thread.start()
            
            # Actualizar color según el usuario
            self.update_colors_for_user(speaker)
            
            # Añadir mensaje
            if len(self.messages) >= self.max_messages:
                self.messages.pop(0)  # Eliminar mensaje más antiguo
            
            self.messages.append({
                'text': f"Profesor: {response}",
                'color': self.bright_blue,
                'time': pygame.time.get_ticks(),
                'duration': self.message_duration
            })
            
        except Exception as e:
            logging.error(f"Error en send_response: {str(e)}")

    def get_font(self, size):
        if size not in self.cached_fonts:
            self.cached_fonts[size] = pygame.font.SysFont('consolas', size)
        return self.cached_fonts[size]

if __name__ == "__main__":
    profesor = Profesor()
    profesor.run()
