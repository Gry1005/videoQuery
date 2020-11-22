from pydub import AudioSegment

voice_dir="E:/cs576/project/data/Data_wav/"
voice1_path=voice_dir+"ads/ads_0.wav"

song = AudioSegment.from_wav(voice1_path)
song.export(voice_dir+"ads_0.mp3", format="mp3")