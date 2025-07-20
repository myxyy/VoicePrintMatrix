from voice_print_matrix.jvs_batch_dataset import JVSBatchDataset
import torchaudio

dataset = JVSBatchDataset(size_ratio=0.01)

waveform, _ = dataset[0]
torchaudio.save(uri='resources/test.wav', src=waveform.reshape(1,-1).cpu().detach(), sample_rate=22050, encoding="PCM_F")
