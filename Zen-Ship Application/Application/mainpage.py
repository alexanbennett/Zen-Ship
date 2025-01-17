from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.animation import Animation
from kivy.core.window import Window
from kivy.uix.screenmanager import ScreenManager, Screen, NoTransition  
from kivy.clock import Clock
from kivy.uix.floatlayout import FloatLayout 
import tkinter as tk
from tkinter import filedialog
from kivy.core.audio import SoundLoader
from StemSep import stemSep
from VS_Array import get_spectrograms
from Chunk_Classification import get_chunk_class
from Reconstruct_Censor import reconstruct_censor_vocals
import numpy as np
import librosa
import soundfile as sf
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from kivy.core.window import Window
from kivy.core.audio import SoundLoader
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.garden.matplotlib import FigureCanvasKivyAgg
from kivy.uix.screenmanager import Screen
import threading
from kivy.uix.progressbar import ProgressBar

global globalfile
class LogoScreen(Screen):
    def on_enter(self):
        layout = BoxLayout(orientation='vertical')
        img = Image(source='Application/logo/Zen-Ship_nb.png', opacity=0, size_hint=(1, .5))
        img.allow_stretch = True
        layout.add_widget(img)

        # opening window
        anim = Animation(opacity=1, duration=4)
        anim.bind(on_complete=self.on_animation_complete)
        anim.start(img)
        self.add_widget(layout)

    def on_animation_complete(self, animation, widget):
        self.manager.current = 'main'

class MainScreen(Screen):
    def on_enter(self):
        layout = FloatLayout()
        logo = Image(source='Application/logo/Zen-Ship_nb.png', size_hint=(None, None), size=(300, 300), pos_hint={'center_x': 0.5, 'top': 1.1})
        layout.add_widget(logo)

        button = Button(text='Upload Song...', size_hint=(None, None), size=(200, 50), pos_hint={'center_x': 0.5, 'center_y': 0.5})
        button.bind(on_press=self.open_file_chooser)
        layout.add_widget(button)

        self.add_widget(layout)

    def open_file_chooser(self, instance):
        global globalfile
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename()
        if file_path:
            print("Selected:", file_path)
            global globalfile
            globalfile = file_path
            self.manager.get_screen('loading')
            self.manager.current = 'loading'  


        root.destroy()
    

class LoadingScreen(Screen):
    global globalfile
    def on_enter(self):
        file_path = globalfile
        layout = BoxLayout(orientation='vertical')
        self.logo = Image(source='Application/logo/Zen-Ship_nb.png', size_hint=(None, None), size=(300, 300), pos_hint={'center_x': 0.5, 'top': 1.1})
        layout.add_widget(self.logo)
        self.status_label = Label(text='Processing...')
        layout.add_widget(self.status_label)

        self.progress_bar = ProgressBar(max=100)  
        layout.add_widget(self.progress_bar)

        self.add_widget(layout) 
        threading.Thread(target=self.process_in_background, args=(file_path,)).start()

    def update_progress(self, value):
        def update(dt):
            self.progress_bar.value = value
        Clock.schedule_once(update)

    def process_in_background(self, file_path):
        # Loading window
        self.update_status('Analyzing...')
        self.update_progress(25)
        sample_rate, instrumental, vocals = stemSep(file_path)
        Clock.schedule_once(lambda dt: self.update_status('Generating spectrograms...'))
        self.update_progress(50)
        au_to_spec = get_spectrograms(vocals, sample_rate)
        Clock.schedule_once(lambda dt: self.update_status('Classifying spoken words...'))
        self.update_progress(75)
        chunk_class = get_chunk_class(au_to_spec)
        Clock.schedule_once(lambda dt: self.update_status('Reconstructing vocals...'))
        self.update_progress(100)
        recon_vocals = reconstruct_censor_vocals(chunk_class, sample_rate)
        Clock.schedule_once(lambda dt: self.update_status(' '))
        
        Clock.schedule_once(lambda dt: self.finish_processing(file_path, recon_vocals, instrumental, sample_rate))

    def update_status(self, text):
        Clock.schedule_once(lambda dt: setattr(self.status_label, 'text', text))
        

    def finish_processing(self, file_path, recon_vocals, instrumental, sample_rate):

        if len(recon_vocals) > len(instrumental):
            print("mismatch in size. Vocals: ", len(recon_vocals), "instrumental: ", len(instrumental))
            instrumental = np.pad(instrumental, (0, len(recon_vocals) - len(instrumental)), 'constant')

        elif len(instrumental) > len(recon_vocals):
            print("mismatch in size. Vocals: ", len(recon_vocals), "instrumental: ", len(instrumental))
            recon_vocals = np.pad(recon_vocals, (0, len(instrumental) - len(recon_vocals)), 'constant')


       
        if instrumental.ndim == 2 and recon_vocals.ndim == 1:
            recon_vocals_stereo = np.tile(recon_vocals[:, np.newaxis], (1, 2))
        elif instrumental.ndim == 1 and recon_vocals.ndim == 2:
            recon_vocals_mono = np.sum(recon_vocals, axis=1) / 2
            
        else:
            
            recon_vocals_stereo = recon_vocals

        
        censored_song = recon_vocals_stereo + instrumental
        
        sf.write('songs/censo_app.wav', censored_song, sample_rate)
        self.manager.current = 'player'

class PlayerScreen(Screen):
    def __init__(self, **kwargs):
        super(PlayerScreen, self).__init__(**kwargs)
        self.current_sound = None  

    def on_enter(self):
        Window.clearcolor = (0.082, 0.082, 0.082, 1)  
        layout = BoxLayout(orientation='vertical')

        # Add logo at the top
        logo = Image(source='Application/logo/Zen-Ship_nb.png', size_hint=(1, 0.2))
        layout.add_widget(logo)
        censor_another_button = Button(text='Censor Another Song', size_hint=(1, 0.1))
        censor_another_button.bind(on_press=self.go_to_main)
        layout.add_widget(censor_another_button)
        # Load the sound and create the waveform plot for both original and censored song
        original_data, original_rate = sf.read(globalfile)
        censored_data, censored_rate = sf.read('songs/censo_app.wav')
        original_data = np.sum(original_data, axis=1) / 2
        censored_data = np.sum(censored_data, axis=1) / 2
        fig, ax = plt.subplots()
        ax.clear()
        # Plot original song waveform
        ax.plot(np.linspace(0, len(original_data)/original_rate, len(original_data)), original_data,
        label='Original Song', color='red', alpha=0.5) 
        # Plot censored song waveform
        light_green = (0.6, 1, 0.6)
        ax.plot(np.linspace(0, len(censored_data)/censored_rate, len(censored_data)), censored_data, label='Censored Song', color=light_green)
        
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Amplitude')
        ax.legend()
        canvas = FigureCanvasKivyAgg(fig)
        layout.add_widget(canvas)

        #play audio buttons
        play_censored_button = Button(text='Play Censored Song', size_hint=(1, 0.1))
        play_censored_button.bind(on_press=lambda x: self.play_song('songs/censo_app.wav'))
        layout.add_widget(play_censored_button)

        play_original_button = Button(text='Play Original Song', size_hint=(1, 0.1))
        play_original_button.bind(on_press=lambda x: self.play_song(globalfile))
        layout.add_widget(play_original_button)

        self.add_widget(layout)
        stop_button = Button(text='Stop', size_hint=(1, 0.1))
        stop_button.bind(on_press=self.stop_song)
        layout.add_widget(stop_button)

    def play_song(self, file_path):
        
        if self.current_sound:
            self.current_sound.stop()
    
        
        self.current_sound = SoundLoader.load(file_path)
        if self.current_sound:
            self.current_sound.play()

    def go_to_main(self, instance):
        self.manager.current = 'main'
        if self.current_sound:
            self.current_sound.stop()
            self.current_sound.unload()
            self.current_sound = None
    def stop_song(self, instance):
       
        if self.current_sound:
            self.current_sound.stop()
            self.current_sound.unload()  
            self.current_sound = None

class MyApp(App):
    def build(self):
        self.title = 'Zen-Ship'
        self.icon = 'Application/logo/Zen-Ship_icon.png'
        Window.clearcolor = (0.082, 0.082, 0.082, 1)
        sm = ScreenManager(transition=NoTransition())
        sm.add_widget(LogoScreen(name='logo'))
        sm.add_widget(MainScreen(name='main'))
        sm.add_widget(LoadingScreen(name='loading'))
        sm.add_widget(PlayerScreen(name='player'))
        return sm

if __name__ == '__main__':
    MyApp().run()
