# Learn a sequence
# compare GRU, CNN and Attention models

This demonstration learns a sequence of text data with GRU, CNN, Att models.
It was designed to compare learning models


### GRU
test_easy: 				min loss: 	0.0888

`tailor_swift_lyrics`:

[elapsed time: 1m 35s, epoch: 1000,  percent complete: 100%, loss: 2.1688]
se it seems to more wall to gorla hreptald and I so cow know
Wo torust you chat you wel fia stow and the norod (won 

Namespace(chunk_len=200, epochs=1000, filename='tailor_swift_lyrics.txt', hidden_size=64, learning_rate=0.001, n_layers=1, pint=16, print_every=25, sequencer='GRU')

Minimum loss during training: 1.5601382446289063




### CNN
test_easy: 				min loss: 	0.0008

`tailor_swift_lyrics`:

[elapsed time: 1m 34s, epoch: 1000,  percent complete: 100%, loss: 1.5704]
had you
Come back a way or beaut 
Mulk and mand that you arous
Dome whorundy down town in the I rassif if same,
Bre

Profiler results:
Namespace(chunk_len=200, epochs=1000, filename='tailor_swift_lyrics.txt', hidden_size=64, learning_rate=0.001, n_layers=1, pint=16, print_every=25, sequencer='CNN')

Minimum loss during training 1.1758484218431555




### Attention: 
test_easy: 				min loss		0.0023

tailor_swift_lyrics: 	min loss: 	1.52
