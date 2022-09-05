from tkinter import *
from tkinter import ttk
from tkinter import filedialog
import tkinter
from tkinter.messagebox import showerror
from tkinter.scrolledtext import ScrolledText
import numpy as np
import Dataset
import Preprocessing
import threading
import sounddevice as sd


labels = {
    0: '/r/ Vibrante alveolar simple o múltiple',
    1: 'Combinación ortográfica <tr> o <dr>',
    2: 'Bien pronunciada',
}
model_path = {
    'cnn': 'C:/Users/sergi/OneDrive/Documentos/Dev/Spanish-Mispronunciations/tensorboard/4_cnn/model',
    'lstm': 'C:/Users/sergi/OneDrive/Documentos/Dev/Spanish-Mispronunciations/tensorboard/1_tags3_lstm_stft/model',
    'custom': '',
}
duration = 4
sample_rate = 44100
tf_response = {'data': {}, 'msg': ''}


class App(ttk.Frame):
    def __init__(self, root):
        super().__init__(root)
        root.title('Identificación de errores frecuentes en el idioma español')

        #   Model
        self.model, self.dnn, self.model_path = Dataset.load_model(model_path['cnn'])

        #   Menu - File
        self.menu_file = ttk.Menubutton(root, text='File')
        self.menu_file.menu = Menu(self.menu_file, tearoff=False)
        self.menu_file['menu'] = self.menu_file.menu
        self.menu_file.menu.add_command(label='Open...', command=self.open_files)
        self.menu_file.menu.add_separator()
        self.menu_file.menu.add_command(label='Exit', command=root.quit)
        self.menu_file.grid(row=0, column=0)

        #   Menu - Model
        self.menu_model = ttk.Menubutton(root, text='Model')
        self.menu_model.menu = Menu(self.menu_model, tearoff=False)
        self.menu_model['menu'] = self.menu_model.menu
        self.menu_model.rb = tkinter.StringVar()
        self.menu_model.rb.set(model_path['cnn'])
        for model in model_path:
            if model == 'custom':
                self.menu_model.menu.add_separator()

            self.menu_model.menu.add_radiobutton(
                label=model.upper(),
                variable=self.menu_model.rb,
                value=model_path[model],
                command=self.open_model,
            )
        self.menu_model.grid(row=0, column=1)

        #   Progress Bar
        self.progress_bar = ttk.Progressbar(root, orient='horizontal', mode='indeterminate', length=40)
        self.progress_bar.grid(row=0, column=59)

        #   Record
        self.button_record = ttk.Button(root, width=60, text='Record', command=self.record_audio)
        self.button_record.grid(row=1, column=0, columnspan=60)

        #   Terminal
        self.terminal = ScrolledText(root, width=60, height=10)
        self.terminal.grid(row=2, column=0, columnspan=60)
        self.terminal.configure(state='disabled')

    def create_thread(self, target=None, args=()):
        self.progress_bar.start(10)
        thread = threading.Thread(target=target, args=args)
        thread.start()
        self.monitor_thread(thread)

    def load_model(self, path):
        global tf_response
        self.model, self.dnn, self.model_path = Dataset.load_model(path)
        tf_response['msg'] += self.dnn.upper() + ' model loaded\n'
        return

    def monitor_thread(self, thread):
        global tf_response

        if thread.is_alive():
            self.after(20, lambda: self.monitor_thread(thread))
        else:
            thread.join()
            if tf_response['msg']:
                self.print_terminal(tf_response['msg'])
            self.progress_bar.stop()

    def open_files(self):
        try:
            filenames = filedialog.askopenfilenames(title='Select audio files', filetypes=[('wav files', '.wav')])
            if filenames:
                self.create_thread(target=self.predict_files, args=(filenames,))
        except Exception as ex:
            showerror(title='Error', message='Unexpected error: ' + str(ex))
            self.progress_bar.stop()
        return

    def open_model(self):
        try:
            path = self.menu_model.rb.get()
            if path == 'CUSTOM':
                path = filedialog.askdirectory(title='Select Model')
            if path and path != self.model_path:
                    self.create_thread(target=self.load_model, args=(path,))
            else:
                for key, value in model_path.items():
                    if path == value:
                        self.menu_model.rb.set(model_path[key])
                        break
        except Exception as ex:
            showerror(title='Error', message='Unexpected error: ' + str(ex))
        return

    def predict_files(self, filenames, audio=[]):
        global tf_response

        if filenames:
            dataset = Dataset.audio_as_dataset(filenames)
        elif len(audio):
            dataset = Dataset.audio_as_dataset(audio=audio)

        dataset = Preprocessing.audio_as_dataset(dataset, self.dnn)
        classifications = np.argmax(self.model.predict(dataset), axis=-1)

        if filenames:
            for index in range(len(filenames)):
                tf_response['msg'] += (
                    '- ' + filenames[index].rsplit('/')[-1] + ' classified as: ' + labels[classifications[index]] + '\n'
                )
        elif len(audio):
            tf_response['msg'] += '- Audio classified as: ' + labels[classifications[0]] + '\n'
        return

    def print_terminal(self, msg='', op=tkinter.INSERT):
        global tf_response
        self.terminal.configure(state="normal")
        self.terminal.insert(op, msg)
        self.terminal.configure(state="disabled")
        if tf_response['msg']:
            tf_response = {'data': {}, 'msg': ''}

    def record_audio(self):
        try:
            audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
            sd.wait()
            if len(audio):
                # sd.play(audio, sample_rate)
                self.create_thread(
                    target=self.predict_files,
                    args=(
                        None,
                        audio,
                    ),
                )
        except Exception as ex:
            showerror(title='Error', message='Unexpected error: ' + str(ex))
            self.progress_bar.stop()
        return


if __name__ == '__main__':
    app = App(Tk())
    app.mainloop()
