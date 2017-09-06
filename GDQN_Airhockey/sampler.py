from utils.ReplayBuffer import ReplayBufferSampler as RBS

sampler = RBS(8,1,400000)
a = sampler.LoadBuffer('hockey_extended_bigRB/Replay_buffer')
print 'Buffer Loaded: ', a
sampler.SampleBuffer(10000, 'validation_buffer')
