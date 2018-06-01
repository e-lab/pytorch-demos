# Learn a sequence: compare GRU, CNN and Attention models

This demonstration learns a sequence of text data with GRU, CNN, Att models.
It was designed to compare learning models

As you can see below CNN do better than GRU in memorizing these sequences, Attention networks are superior and ~2x CNN size in params/computation.


### GRU

`test_easy`:

[elapsed time: 1m 30s, epoch: 1000,  percent complete: 100%, loss: 0.1088]
 livin'
Yes, I'm workin' home at five o'clock
And I take myself out a nice, cold bee
A lot better than I think I am

Namespace(chunk_len=200, epochs=1000, filename='test_easy.txt', hidden_size=64, learning_rate=0.001, n_layers=1, pint=16, print_every=25, sequencer='GRU')

Minimum loss during training: 0.057


`tailor_swift_lyrics`:

[elapsed time: 1m 35s, epoch: 1000,  percent complete: 100%, loss: 2.1688]
se it seems to more wall to gorla hreptald and I so cow know
Wo torust you chat you wel fia stow and the norod (won 

Namespace(chunk_len=200, epochs=1000, filename='tailor_swift_lyrics.txt', hidden_size=64, learning_rate=0.001, n_layers=1, pint=16, print_every=25, sequencer='GRU')

Minimum loss during training: 1.56




### CNN

`test_easy`:

[elapsed time: 1m 40s, epoch: 1000,  percent complete: 100%, loss: 0.0198]
hey call me the working man

They call me the working man
I guess that's why they call me
They call me the working m

Namespace(chunk_len=200, epochs=1000, filename='test_easy.txt', hidden_size=64, learning_rate=0.001, n_layers=1, pint=16, print_every=25, sequencer='CNN')

Minimum loss during training: 0.00030



`tailor_swift_lyrics`:

[elapsed time: 1m 34s, epoch: 1000,  percent complete: 100%, loss: 1.5704]
had you
Come back a way or beaut 
Mulk and mand that you arous
Dome whorundy down town in the I rassif if same,
Bre

Namespace(chunk_len=200, epochs=1000, filename='tailor_swift_lyrics.txt', hidden_size=64, learning_rate=0.001, n_layers=1, pint=16, print_every=25, sequencer='CNN')

Minimum loss during training 1.17




### Attention: 

`test_easy`:

[elapsed time: 2m 45s, epoch: 1000,  percent complete: 100%, loss: 0.0111]
self out a nice, cold beer
Always seem to be wonderin'
Why there's nothin' goin' down here

[Chorus]

Well, they cal

Namespace(chunk_len=200, epochs=1000, filename='test_easy.txt', hidden_size=64, learning_rate=0.001, n_layers=1, pint=16, print_every=25, sequencer='Att')

**Minimum loss during training: 0.00026**


`tailor_swift_lyrics`:
[elapsed time: 2m 34s, epoch: 1000,  percent complete: 100%, loss: 2.0854]
d the message throme come
Do sawing sto shide I'm got come
I ameal sveay th albid the start of whe ond tright
And I 

Namespace(chunk_len=200, epochs=1000, filename='tailor_swift_lyrics.txt', hidden_size=64, learning_rate=0.001, n_layers=1, pint=16, print_every=25, sequencer='Att')

**Minimum loss during training: 0.89**




### Attention - hidden size 32 (~same comp as other nets):

`test_easy`:

[elapsed time: 1m 57s, epoch: 1000,  percent complete: 100%, loss: 0.0181]
nothin' goin' down here

[Chorus]

Well, they call me the working man
I guess that's what I am


"Working Man"

I ge
Namespace(chunk_len=200, epochs=1000, filename='test_easy.txt', hidden_size=32, learning_rate=0.001, n_layers=1, pint=16, print_every=25, sequencer='Att')

Minimum loss during training: 0.00077


`tailor_swift_lyrics`:

[elapsed time: 2m 1s, epoch: 1000,  percent complete: 100%, loss: 2.2359]
e birthday boy weem douget you on, I know
's waek you tore saating ameI
Ye heac
Wre w'st goke stow

And me neky
And 

Namespace(chunk_len=200, epochs=1000, filename='tailor_swift_lyrics.txt', hidden_size=32, learning_rate=0.001, n_layers=1, pint=16, print_every=25, sequencer='Att')

Minimum loss during training: 1.64