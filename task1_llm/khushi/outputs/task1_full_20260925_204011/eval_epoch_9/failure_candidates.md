# Task 1 failure-case candidates: task1_full_20260925_204011, epoch 9 checkpoint

Samples ranked by objective per-sample statistics only. No failure types are assigned here: read the samples (all of them are in generations.txt), then choose and categorize three cases yourself.

## Highest character-level repeated 4-gram rate (primary diversity metric)

**Sample 21** (prompt 'Lily and Ben'): `repeated_4gram_rate_char` = 0.244; char repeated 4-gram rate 0.244, word repeated 4-gram rate 0.000, unseen-word share 0.000, non-ASCII chars 0

```text
Lily and Ben were playing in the park. They dropped their house and ran a car. They were playing a game. They loved to write the train. They sang and laughed. They looked at each other. They were sad.

They had a
```

**Sample 2** (prompt 'Once upon a time'): `repeated_4gram_rate_char` = 0.239; char repeated 4-gram rate 0.239, word repeated 4-gram rate 0.025, unseen-word share 0.047, non-ASCII chars 0
Unseen words: folden, sin

```text
Once upon a time there was a big boy who wanted to play outside. The boy was so happy and he was so happy and wished it. He looked deaf to his mom and dad opened the folden. 

The photo wanted to play that he was sin
```

**Sample 8** (prompt 'Once upon a time'): `repeated_4gram_rate_char` = 0.239; char repeated 4-gram rate 0.239, word repeated 4-gram rate 0.051, unseen-word share 0.000, non-ASCII chars 0

```text
Once upon a time, there was a little girl named Lily. She loved to play in the park. One day, a little girl named Lily went to the park. She made a loud noise. She was so excited for a magic. She didn't like the food
```

## Highest word-level repeated 4-gram rate (secondary diagnostic)

**Sample 8** (prompt 'Once upon a time'): `repeated_4gram_rate_word` = 0.051; char repeated 4-gram rate 0.239, word repeated 4-gram rate 0.051, unseen-word share 0.000, non-ASCII chars 0

```text
Once upon a time, there was a little girl named Lily. She loved to play in the park. One day, a little girl named Lily went to the park. She made a loud noise. She was so excited for a magic. She didn't like the food
```

**Sample 0** (prompt 'Once upon a time'): `repeated_4gram_rate_word` = 0.025; char repeated 4-gram rate 0.152, word repeated 4-gram rate 0.025, unseen-word share 0.000, non-ASCII chars 0

```text
Once upon a time, there was a little girl named Lily. She had a big box to the mountain. One day, her mom saw a big box to sparkle and the spider. They also learned that her name cleaning the boy and had a send and a
```

**Sample 2** (prompt 'Once upon a time'): `repeated_4gram_rate_word` = 0.025; char repeated 4-gram rate 0.239, word repeated 4-gram rate 0.025, unseen-word share 0.047, non-ASCII chars 0
Unseen words: folden, sin

```text
Once upon a time there was a big boy who wanted to play outside. The boy was so happy and he was so happy and wished it. He looked deaf to his mom and dad opened the folden. 

The photo wanted to play that he was sin
```

## Highest share of words never seen in the training split (secondary diagnostic)

Caveat: every sample stops at exactly 200 new characters, so its last word can be cut off mid-word, and such fragments count as unseen words. This inflates this secondary diagnostic; do not use it as a primary failure-selection criterion.

**Sample 32** (prompt 'The little bird'): `unseen_word_share` = 0.049; char repeated 4-gram rate 0.112, word repeated 4-gram rate 0.000, unseen-word share 0.049, non-ASCII chars 0
Unseen words: hou, strucks

```text
The little bird was so excited to act the forest, and the bird was so surprised. The bird flew strucks were almost and happy that he would find a dog has to get his collect shone. He laughed and got there on his hou
```

**Sample 2** (prompt 'Once upon a time'): `unseen_word_share` = 0.047; char repeated 4-gram rate 0.239, word repeated 4-gram rate 0.025, unseen-word share 0.047, non-ASCII chars 0
Unseen words: folden, sin

```text
Once upon a time there was a big boy who wanted to play outside. The boy was so happy and he was so happy and wished it. He looked deaf to his mom and dad opened the folden. 

The photo wanted to play that he was sin
```

**Sample 16** (prompt 'One day,'): `unseen_word_share` = 0.045; char repeated 4-gram rate 0.127, word repeated 4-gram rate 0.000, unseen-word share 0.045, non-ASCII chars 0
Unseen words: bi, ittle

```text
One day, a little girl named Lily went outside to play with her toy dolls. She did not like them all the park. One day, Lily's mom asked her to Sam what was in the sky with a hat ittle boy. She said it was bi
```

## Most characters outside printable ASCII

All generated samples contained only printable ASCII characters; no candidates identified for this criterion.

