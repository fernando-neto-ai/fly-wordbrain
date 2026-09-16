# Rank-64 decoder quality: G versus rank-128 E

Sequential MPS inference on idle macm3. Both retained selectors replay all 100 validation stories / 21,874 targets. The reserved test is untouched.

| Arm | Selector | Update | Training targets | CE | Accuracy |
|---|---|---:|---:|---:|---:|
| E32rank128fixed | minimum_validation_ce | 13,700 | 2,689,179 | 3.14970 | 36.482% |
| E32rank128fixed | maximum_validation_accuracy | 14,300 | 2,806,777 | 3.15991 | 36.582% |
| G32rank64fixed | minimum_validation_ce | 16,600 | 3,257,286 | 3.09371 | 36.367% |
| G32rank64fixed | maximum_validation_accuracy | 15,552 | 3,052,348 | 3.10345 | 36.889% |

## Retained winners: G minus E

- minimum_validation_ce: CE -0.05600; accuracy -0.114 percentage points.
- maximum_validation_accuracy: CE -0.05646; accuracy +0.306 percentage points.

## Common budget: 15,900 updates / 3,120,358 training targets

| Update | E CE | G CE | E accuracy | G accuracy |
|---:|---:|---:|---:|---:|
| 0 | 6.95394 | 6.94781 | 0.073% | 0.091% |
| 1,000 | 4.67895 | 4.75161 | 19.356% | 18.533% |
| 2,000 | 4.02793 | 4.12482 | 24.298% | 23.265% |
| 3,000 | 3.72905 | 3.76719 | 27.754% | 27.352% |
| 4,000 | 3.51931 | 3.58852 | 30.584% | 29.542% |
| 5,000 | 3.42047 | 3.47548 | 32.381% | 31.407% |
| 6,000 | 3.30841 | 3.38569 | 33.643% | 32.084% |
| 7,000 | 3.28933 | 3.32612 | 34.018% | 32.829% |
| 8,000 | 3.21817 | 3.24938 | 35.028% | 34.402% |
| 9,000 | 3.22033 | 3.23210 | 35.138% | 34.840% |
| 10,000 | 3.19624 | 3.19293 | 35.234% | 34.950% |
| 11,000 | 3.17862 | 3.17453 | 35.928% | 35.608% |
| 12,000 | 3.17715 | 3.15275 | 36.043% | 35.732% |
| 13,000 | 3.16430 | 3.13063 | 36.258% | 36.089% |
| 14,000 | 3.17723 | 3.13936 | 36.290% | 36.006% |
| 15,000 | 3.18379 | 3.12110 | 36.198% | 36.171% |
| 15,900 | 3.21513 | 3.13186 | 35.805% | 35.896% |

Retained winners have their actual budgets above. The JSON also records both best selectors through the common budget; no unavailable historical checkpoint text is inferred.

## Unedited greedy continuations

Same six prompts, fresh cache, one BOS, EOS stopping, at most 80 new BPE tokens. Fluency and overlap are descriptive, not a held-out generalization claim.

### 1. Once upon a time, there was a

**E32rank128fixed / minimum_validation_ce**

 little girl named Lily. She loved to play with her toys and run around. One day, she saw a big, she saw a big tree. She wanted to show her mom, "Mommy, can I do it. Her mom said yes, and said, "I love you too, my little tall girl." 

Lily and said, "No, I don

Longest contiguous training overlap: 13 whitespace words. Full text is a normalized training prefix: False.

**E32rank128fixed / maximum_validation_accuracy**

 little girl named Lily. She loved to play outside and play with her friends. One day, Lily's mom asked her to help her. Her mom said yes, Lily didn't want to to do. She said to herself.

Lily said, "I can help you to play with her toys. Her mom said, "Don't worry, Lily. We can

Longest contiguous training overlap: 10 whitespace words. Full text is a normalized training prefix: False.

**G32rank64fixed / minimum_validation_ce**

 little girl named Lily. She loved to play with her toys and play with her toys. One day, Lily's mom asked her to help her Mom. Lily was very excited and she wanted to play with her toys. Lily was so excited to see her hands. She asked her mom if she could helpwards the trip. She was so excited to explore the world

Longest contiguous training overlap: 12 whitespace words. Full text is a normalized training prefix: False.

**G32rank64fixed / maximum_validation_accuracy**

 little girl named Lily. She loved to play outside and explore the world around her. One day, she went to the park with her mom. She saw a big, and saw a big, round. It was a big tree. The little girl was very happy. She had a big, but she had made it.

One day, she decided to go on a big smile

Longest contiguous training overlap: 20 whitespace words. Full text is a normalized training prefix: False.

### 2. One day, a little girl named Lily found a needle

**E32rank128fixed / minimum_validation_ce**

 in the garden. She saw a big, rusle in the water. She was very excited and wanted to buy it. She wanted to buy a big bite and grabbed her toys. 

Lily's mom said she'd for her mom said, "Do you, my girl and playget again. Weeze, Lily's mom said

Longest contiguous training overlap: 6 whitespace words. Full text is a normalized training prefix: False.

**E32rank128fixed / maximum_validation_accuracy**

 in the garden. She saw a big, colorful cat. She was very tiny and curious.

But when she was very trawl, "What's you welcome?"

Her mom said, "No, Lily. I want to play with you, but it didn't want to behave to make a net.

Longest contiguous training overlap: 6 whitespace words. Full text is a normalized training prefix: False.

**G32rank64fixed / minimum_validation_ce**

 in the park. She saw a big, scary was very happy and wanted to play with her toys. She said, "I want to be careful, but it is too! I can be careful to the park, they can play with the balloon. They both was so happy and said, "I can play with the balloon, they bothst my tired."

Longest contiguous training overlap: 6 whitespace words. Full text is a normalized training prefix: False.

**G32rank64fixed / maximum_validation_accuracy**

 in the forest. She was very excited. She wanted to go outside and play with her friends. She was so excited. She wanted to help her mom. She said, "I'm sour of the meet is that?" The little girl said, "I'm you help you, Lily. We can play with the mygest."

The little girl smiled and said,

Longest contiguous training overlap: 7 whitespace words. Full text is a normalized training prefix: False.

### 3. Tom and his dog

**E32rank128fixed / minimum_validation_ce**

 were very important. He loved to play with his friends. One day, he found a big, pink that he was very lighty and said, "Do you, I don't know". You'd go?"

"I don't want to my mountain. What's friends?" said, "I'm counting, I

Longest contiguous training overlap: 7 whitespace words. Full text is a normalized training prefix: False.

**E32rank128fixed / maximum_validation_accuracy**

 were walking down. They were playing with a big bow. They both had a great time together. They were so happy and they were all very happy. They thanked the other children, and they were very happy. They learned that it's important to always behave and make them to make it again.

Longest contiguous training overlap: 5 whitespace words. Full text is a normalized training prefix: False.

**G32rank64fixed / minimum_validation_ce**

 were walking through the forest, they saw a big, a big tree. The boy was so excited. He wanted to go on the train and it was too quick. He wanted to find a way to play with his toys. He looked around and saw a big tree. He was so excited! He wanted to get a new friend, but he couldn't find his way to the

Longest contiguous training overlap: 7 whitespace words. Full text is a normalized training prefix: False.

**G32rank64fixed / maximum_validation_accuracy**

 were walking through the forest. He was going to the park and he was very excited. He wanted to try it.

John looked around the house and saw a big, rush. He was so excited! He wanted to get it up the roof. He asked the rabbit to make it.

The little girl was so excited. He wanted to try and

Longest contiguous training overlap: 6 whitespace words. Full text is a normalized training prefix: False.

### 4. Sara put the red ball inside the box. Later,

**E32rank128fixed / minimum_validation_ce**

 the birds were very sad. They had a big coneble. It was a big and yummy. 

They was very happy and said, "You're welcome! Can I have a new friend."

The little girl was very happy and said, "I love, I will make my move." 

Lily was so happy and

Longest contiguous training overlap: 7 whitespace words. Full text is a normalized training prefix: False.

**E32rank128fixed / maximum_validation_accuracy**

 Daddy said he didn't like to the park.

"I'm soft, Mommy," said Mary. "May, I'll be a friend!"

Jack said, "Yes, you can help you!"

The little girl smiled and said, "If you helpse!" Her mom smiled and said, "Yes, let's

Longest contiguous training overlap: 7 whitespace words. Full text is a normalized training prefix: False.

**G32rank64fixed / minimum_validation_ce**

 the monster was so happy. He wanted to find a way to play with his toys. He saw a big, filled with joy and he was so happy.

The little girl was so excited to have a new friend to play with. She was so happy to have a special party. She knew that she had a wonderful time, she was so

Longest contiguous training overlap: 8 whitespace words. Full text is a normalized training prefix: False.

**G32rank64fixed / maximum_validation_accuracy**

 the dog were very happy. They were so excited. He wanted to go back and the window.

After a while, the tree was walking in the park. He saw a big tree and saw a big tree. The bear was very happy. He had a big bedroom and he was very happy.

The little boy was so excited. He wanted to know

Longest contiguous training overlap: 7 whitespace words. Full text is a normalized training prefix: False.

### 5. The little bird was afraid of the rain. Its mother

**E32rank128fixed / minimum_validation_ce**

 was very happy and could have it.

One day, a little boy named Tim came to a big smile on his face. He was very sad and he wanted to see what it was. He said, "I wra you want to catch a biteker. You can youngly threeend the world's face, I can take you to look

Longest contiguous training overlap: 9 whitespace words. Full text is a normalized training prefix: False.

**E32rank128fixed / maximum_validation_accuracy**

 and the rabbit was very happy.

One day, the little girl was very berful. He wanted to help the birds. She ran around and saw a big, river.

But then, the rabbit decided to take a bath. He was so excited to see the race, but it was too fast. He decided to take a nap.

Longest contiguous training overlap: 6 whitespace words. Full text is a normalized training prefix: False.

**G32rank64fixed / minimum_validation_ce**

 was very happy and excited. He was always being to celebrated the sun.

One day, a little boy named Timmy went to the park. He saw a big tree on a big tree. The bear was so excited he wanted to play with his toys. He put it on the bed and he was so happy. He said, "No,

Longest contiguous training overlap: 7 whitespace words. Full text is a normalized training prefix: False.

**G32rank64fixed / maximum_validation_accuracy**

 was very sad, but he wanted to find something. He wanted to get some new treats.

Suddenly, the little girl named Sally saw a big, let's go. She wanted to play with it, but it was too small. She wanted to get closer, so she asked her mom if she could help.

The little girl was so excited that she

Longest contiguous training overlap: 9 whitespace words. Full text is a normalized training prefix: False.

### 6. Ben wanted to share his cake, but

**E32rank128fixed / minimum_validation_ce**

 he wanted to see what was too. He wanted to get some angry and started to feel als. He looked around and saw a big, strongige. He wanted to build something so he asked his mom, "What are you donâ€™t need to see he use. He wanted to know what he was.

When he got to the park,

Longest contiguous training overlap: 7 whitespace words. Full text is a normalized training prefix: False.

**E32rank128fixed / maximum_validation_accuracy**

 he wanted to play. He was so excited to see the vase, he found something new. He was a pief of the rope. He was so excited to his mouse and said he could help. He was so excited to see the modest.

He started to walk. He was very excited and wanted to get some more.

The

Longest contiguous training overlap: 6 whitespace words. Full text is a normalized training prefix: False.

**G32rank64fixed / minimum_validation_ce**

 he was very lady. He was so excited to see the most beautiful and held. He was a very sleepy. He was so excited to see him. He was so excited to see the mostle and ran to the park. He was so happy and he had a wonderful time. He knew he had to the park all his days.

Longest contiguous training overlap: 6 whitespace words. Full text is a normalized training prefix: False.

**G32rank64fixed / maximum_validation_accuracy**

 he didn't know what to do. He wanted to take a bit of the ridd.

One day, the boy named Tim saw a big, he saw a big tree. He wanted to play with his friends. He saw a big, but he couldn't find anything.

The little boy was so excited. He wanted to try it, but he couldn't

Longest contiguous training overlap: 7 whitespace words. Full text is a normalized training prefix: False.

## Per-neuron parameters

| Selector | Parameter | G−E RMS | G−E max absolute | Changed fraction |
|---|---|---:|---:|---:|
| minimum_validation_ce | brain.gain | 0.260185 | 2.04339 | 100.000% |
| minimum_validation_ce | brain.rec_gain | 10.4606 | 301.33 | 100.000% |
| minimum_validation_ce | brain.bias | 0.102559 | 0.983499 | 100.000% |
| minimum_validation_ce | gain_times_rec_gain | 172.736 | 11013.6 | 100.000% |
| maximum_validation_accuracy | brain.gain | 0.257533 | 1.93178 | 100.000% |
| maximum_validation_accuracy | brain.rec_gain | 4.59583 | 132.271 | 100.000% |
| maximum_validation_accuracy | brain.bias | 0.101924 | 0.970967 | 100.000% |
| maximum_validation_accuracy | gain_times_rec_gain | 167.574 | 11595.2 | 100.000% |

The JSON contains each winner’s deltas from its hash-verified initialization. Canonical edge values, connectivity and interfaces are exactly fixed; neuron gains/biases are trainable. These differences do not establish anatomical advantage or compensation causality. One seed, different decoder shapes/RNG consumption, checkpoint-selection validation, and potentially unequal budgets limit interpretation.
