# Reduced-encoder quality assessment: A versus B

Inference only on macm3. Both retained selectors replay the complete validation set; the reserved test is untouched.
Retained winners may have unequal training budgets. Common-update rows below compare logged scores, not unavailable historical checkpoint weights.

| Arm | Selector | Update | CE | Accuracy | Replay |
|---|---|---:|---:|---:|---|
| A128fixed | minimum_validation_ce | 3,600 | 4.72828 | 30.488% | passed |
| A128fixed | maximum_validation_accuracy | 12,200 | 5.54698 | 32.898% | passed |
| B32fixed | minimum_validation_ce | 9,700 | 4.51251 | 31.526% | passed |
| B32fixed | maximum_validation_accuracy | 15,200 | 4.62185 | 33.314% | passed |

## B minus A at retained winners

- minimum_validation_ce: ΔCE -0.21577; accuracy +1.038 pp; practical review flag: False.
- maximum_validation_accuracy: ΔCE -0.92513; accuracy +0.416 pp; practical review flag: False.

The review flags are CE loss greater than 0.10 or accuracy loss greater than 1 pp. These are practical flags, not significance tests. Text coherence still requires inspection.

## All fixed greedy continuations

Same six prompts, fresh cache and one BOS per prompt, greedy decoding, EOS stopping and at most 80 new BPE tokens. Outputs are unedited.

### 1. Once upon a time, there was a

**A128fixed / minimum_validation_ce**

 little boy named Timmy. Timmy loved to play with his ball with his favorite black. One day, Timmy's mom told him to the wrack. Timmy saw Timmy to his arms and listen to his mom and said, "Mommy, I we can we do you!" The mor morning?" asked him if you can pour your pond?"


Longest contiguous training overlap: 10 whitespace words; full generated text is a normalized training prefix: False.

**A128fixed / maximum_validation_accuracy**

 boy named Timmy. Timmy loved to play outside in the sun. One day, Timmy found a big card with a triang he had never seen before. He even rough the animals were very best friend, a wrangled. 

Finally, the shark had banke up and saw the rolls very hungry. He

Longest contiguous training overlap: 16 whitespace words; full generated text is a normalized training prefix: False.

**B32fixed / minimum_validation_ce**

 little girl named Lily. She loved to play with her toys all day long. But one day, her mommy asked her to help tidy up her room. It was a big task! She was so happy that she hugged her mommy again and said, "No, you must! The about meon said, "I!" Nor, I said, "N

Longest contiguous training overlap: 31 whitespace words; full generated text is a normalized training prefix: False.

**B32fixed / maximum_validation_accuracy**

 little girl named Lily. She loved to play with her toys on the table. One day, Lily's mom said, "Lily, it's time to clean the table." Lily replied, "Okay, mommy!" 

Lily wiped the table with a clean cloth. Suddenly, the ice cream began to melt! She quickly

Longest contiguous training overlap: 39 whitespace words; full generated text is a normalized training prefix: False.

### 2. One day, a little girl named Lily found a needle

**A128fixed / minimum_validation_ce**

 with her. She was so happy to see him. She asked her home if her and had a great time. She would at herself and she would always be a friend. One day, her mommy gave her appoon to the vend with her family and her dad. She was happy to have a safe ball with her and yummy. One day, Timmy's mom told

Longest contiguous training overlap: 6 whitespace words; full generated text is a normalized training prefix: False.

**A128fixed / maximum_validation_accuracy**

 in her room. She was so excited to see it lady. She would put the lovely in the biggot.

The little boy was very sad. He was happy to the par the beg, and the little girl had a wonderful why obed. She was so elephant's mom and she gave him a big hug. T

Longest contiguous training overlap: 7 whitespace words; full generated text is a normalized training prefix: False.

**B32fixed / minimum_validation_ce**

ss her room. She was so excited to put on her face, and she went on her wide.

Suddenly, she saw a flower lonely and wanted to pick one but she knew she was so excited. She quickly picked it up and opened it into the lake.

Hel it was as she so she could get it again. Then she show

Longest contiguous training overlap: 5 whitespace words; full generated text is a normalized training prefix: False.

**B32fixed / maximum_validation_accuracy**

 in her room. She knew it was difficult to her because she was. She took a bite and smile on her face. One day, it was very rient. He was singing on the brownchable.

John and his mom were walking and they were him. Daisy was very happy. She was the word

Longest contiguous training overlap: 9 whitespace words; full generated text is a normalized training prefix: False.

### 3. Tom and his dog

**A128fixed / minimum_validation_ce**

 were are morning. An the woke up the morning, the guard was always so happy.

The girl had to the garage was very thankful. Her like the wonderful feeling wet and had a great time.

Longest contiguous training overlap: 5 whitespace words; full generated text is a normalized training prefix: False.

**A128fixed / maximum_validation_accuracy**

 were lots of dinoes. The little boy was filled with joy. One of the jungle and saw the rolls. It wanted to help him. The bird was so happy to see how that he was first, but he kept the gerher. He eventually he found a beautiful fish.

When he finally

Longest contiguous training overlap: 5 whitespace words; full generated text is a normalized training prefix: False.

**B32fixed / minimum_validation_ce**

 were very happy.

His morning's important to be careful when him tall as he was appy. He thanked the teacher for his ve it way to him and his cold outside. He was so happy to see him he had his newcket. He hugged his way home, And he was happy to be careful.

Longest contiguous training overlap: 6 whitespace words; full generated text is a normalized training prefix: False.

**B32fixed / maximum_validation_accuracy**

 were walking down the street. He was glad and he was always looking for something. He couldn't figure out his treat. He was having so much fun.

The rocket and the filled the birdcage. He saused that the same place, so he slowly. It was very horn special.

Longest contiguous training overlap: 6 whitespace words; full generated text is a normalized training prefix: False.

### 4. Sara put the red ball inside the box. Later,

**A128fixed / minimum_validation_ce**

 the wind began to molend. One day, the little girl went to the old with his favorite. He was tased with her and plusage to the trand to listen to each her parents. 

The morning, the girl had listened to her what. She hugged her tightly. She was very happy and her

Longest contiguous training overlap: 8 whitespace words; full generated text is a normalized training prefix: False.

**A128fixed / maximum_validation_accuracy**

 Sarah was sore that she could have the penny. Sarah was so happy that she could help the penny. She went to the match for help.

When the per was finally, they got to go outside. She looked up at the sky and saw a beautiful rainbow. She soon forgot her such

Longest contiguous training overlap: 14 whitespace words; full generated text is a normalized training prefix: False.

**B32fixed / minimum_validation_ce**

 it was apping from the garden. The little girl had so much fun with the butterfly in the sky. It was so happy that she wanted to ear again.

The pairs stay she said, "That's myleash."

"Don't worry," said Bongo. "I will help you." Bongo picked

Longest contiguous training overlap: 10 whitespace words; full generated text is a normalized training prefix: False.

**B32fixed / maximum_validation_accuracy**

 and felt so he went for a place. She pointed to win the bird. The little bird was singing.

Jack and Jane got closer to the rubber duckymbed. His dad said, "Let's genc toude.

"Hio can Frank, and the fill g

Longest contiguous training overlap: 4 whitespace words; full generated text is a normalized training prefix: False.

### 5. The little bird was afraid of the rain. Its mother

**A128fixed / minimum_validation_ce**

s and gre can pull the birds and wet. The girl was so happy and they had a new friendly.

The cat were so happy to be home with him. She stayed him for the amazing time and soonromised to list list.

Longest contiguous training overlap: 6 whitespace words; full generated text is a normalized training prefix: False.

**A128fixed / maximum_validation_accuracy**

 for him. 
Tom decided to take a warm towl came from the hole. So, he went to an older and said, he was a big ber. He was worked hard and make it was clean again. He was so happy!

Longest contiguous training overlap: 4 whitespace words; full generated text is a normalized training prefix: False.

**B32fixed / minimum_validation_ce**

 and Jack was very happy. One day, he went to the park with his mom went over to the park. The chea started to him and cold it bet.

Theyat Tomed it started to put their cloth cloth and put on his skich.

After they were the best of laundry and went on a flame, Tom

Longest contiguous training overlap: 7 whitespace words; full generated text is a normalized training prefix: False.

**B32fixed / maximum_validation_accuracy**

 and Jack, but he couldn't find anything. One day, Joe was ingased the tall and he couldn't stop to find his bread. But then, something else to do. He took the knee and the fire, but she was brave. She was so happy and she felt sad. She hugged it and the love.

Longest contiguous training overlap: 6 whitespace words; full generated text is a normalized training prefix: False.

### 6. Ben wanted to share his cake, but

**A128fixed / minimum_validation_ce**

 the voice. The week was very happy and his mommy. There were so happy to see him the being home and yellowlse. The end.

Longest contiguous training overlap: 6 whitespace words; full generated text is a normalized training prefix: False.

**A128fixed / maximum_validation_accuracy**

 the eventually found it. He was so beautiful he said, "This is that the mine!" Soon 
The bears too fircy. He was so happy to have something bifluding.

Longest contiguous training overlap: 7 whitespace words; full generated text is a normalized training prefix: False.

**B32fixed / minimum_validation_ce**

 the best one. Tom wanted to get closer and Daisy, so he asked if it, he wetan it lit him. He asked his dad if he could he heard a so his ner. He was so excited that he had just stayed there for his naughty!

One day, he decided to go for a walk in the garden. He

Longest contiguous training overlap: 11 whitespace words; full generated text is a normalized training prefix: False.

**B32fixed / maximum_validation_accuracy**

 the world. As they looked closely saw lots of loust come. It was compute could gention. Every morning, Tim would find his way to help. He give of the sound.

Longest contiguous training overlap: 3 whitespace words; full generated text is a normalized training prefix: False.

## Common logged updates

| Update | A CE | B CE | ΔCE | A accuracy | B accuracy | Δaccuracy pp |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 7.48656 | 7.32686 | -0.15970 | 0.059% | 0.091% | +0.032 |
| 100 | 7.35872 | 7.21182 | -0.14690 | 8.718% | 8.092% | -0.626 |
| 200 | 6.78038 | 7.02287 | +0.24249 | 13.943% | 11.091% | -2.853 |
| 300 | 6.81020 | 7.30350 | +0.49330 | 12.403% | 10.432% | -1.970 |
| 400 | 6.61622 | 6.96514 | +0.34892 | 14.035% | 11.626% | -2.409 |
| 500 | 5.96856 | 6.37889 | +0.41033 | 17.665% | 14.433% | -3.232 |
| 600 | 6.16410 | 6.72326 | +0.55916 | 17.052% | 13.761% | -3.292 |
| 700 | 6.45866 | 6.77468 | +0.31603 | 17.144% | 15.420% | -1.724 |
| 800 | 6.27734 | 6.66070 | +0.38336 | 16.851% | 13.573% | -3.278 |
| 900 | 5.72897 | 6.33063 | +0.60165 | 22.076% | 16.837% | -5.239 |
| 1,000 | 5.42915 | 6.13263 | +0.70348 | 23.955% | 17.605% | -6.350 |
| 1,100 | 6.23017 | 6.51816 | +0.28799 | 17.752% | 16.257% | -1.495 |
| 1,196 | 5.26985 | 5.82303 | +0.55317 | 23.539% | 17.587% | -5.952 |
| 1,200 | 5.25252 | 5.72774 | +0.47521 | 24.426% | 18.940% | -5.486 |
| 1,300 | 5.67075 | 5.89182 | +0.22106 | 21.080% | 18.200% | -2.880 |
| 1,400 | 5.30855 | 5.97112 | +0.66257 | 23.608% | 17.528% | -6.080 |
| 1,500 | 5.25080 | 5.82911 | +0.57831 | 25.382% | 20.193% | -5.189 |
| 1,600 | 5.07212 | 5.61521 | +0.54309 | 26.355% | 20.975% | -5.381 |
| 1,700 | 5.12130 | 5.89578 | +0.77448 | 25.523% | 18.885% | -6.638 |
| 1,800 | 5.10467 | 5.83691 | +0.73223 | 26.136% | 19.973% | -6.163 |
| 1,900 | 5.25657 | 6.04512 | +0.78855 | 24.957% | 19.722% | -5.235 |
| 2,000 | 4.96934 | 5.57171 | +0.60237 | 26.836% | 22.726% | -4.110 |
| 2,100 | 4.98466 | 5.62613 | +0.64147 | 26.520% | 21.025% | -5.495 |
| 2,200 | 5.02100 | 5.60372 | +0.58272 | 27.210% | 22.986% | -4.224 |
| 2,300 | 4.94164 | 5.68377 | +0.74213 | 27.192% | 20.261% | -6.931 |
| 2,389 | 5.22376 | 5.67735 | +0.45359 | 25.638% | 20.915% | -4.723 |
| 2,400 | 5.07545 | 5.75672 | +0.68127 | 26.945% | 20.705% | -6.240 |
| 2,500 | 4.83670 | 5.24823 | +0.41153 | 27.933% | 23.361% | -4.572 |
| 2,600 | 4.81608 | 5.33433 | +0.51825 | 28.390% | 24.029% | -4.361 |
| 2,700 | 4.90584 | 5.42185 | +0.51601 | 27.942% | 22.963% | -4.979 |
| 2,800 | 4.95967 | 5.49701 | +0.53735 | 27.745% | 23.233% | -4.512 |
| 2,900 | 4.86255 | 5.11933 | +0.25679 | 29.217% | 25.373% | -3.845 |
| 3,000 | 4.97994 | 5.44097 | +0.46103 | 27.649% | 22.552% | -5.097 |
| 3,100 | 5.00020 | 5.34843 | +0.34822 | 27.457% | 23.827% | -3.630 |
| 3,200 | 4.84775 | 5.10476 | +0.25701 | 29.094% | 25.761% | -3.333 |
| 3,300 | 4.86248 | 5.18110 | +0.31862 | 27.700% | 24.989% | -2.711 |
| 3,400 | 4.94784 | 5.11650 | +0.16866 | 28.280% | 24.486% | -3.794 |
| 3,500 | 4.75847 | 5.04467 | +0.28620 | 29.322% | 25.464% | -3.858 |
| 3,587 | 4.81366 | 5.11028 | +0.29662 | 29.057% | 24.641% | -4.416 |
| 3,600 | 4.72828 | 4.94577 | +0.21749 | 30.488% | 26.694% | -3.794 |
| 3,700 | 4.82174 | 5.12113 | +0.29939 | 29.112% | 25.633% | -3.479 |
| 3,800 | 4.89504 | 5.21185 | +0.31681 | 29.016% | 24.847% | -4.169 |
| 3,900 | 4.79030 | 4.84959 | +0.05929 | 29.743% | 27.384% | -2.359 |
| 4,000 | 4.79280 | 4.94898 | +0.15617 | 30.045% | 26.511% | -3.534 |
| 4,100 | 4.99607 | 5.17067 | +0.17460 | 29.405% | 25.267% | -4.137 |
| 4,200 | 4.87436 | 5.55435 | +0.67999 | 30.059% | 20.970% | -9.088 |
| 4,300 | 4.84246 | 4.90348 | +0.06102 | 29.537% | 27.398% | -2.140 |
| 4,400 | 4.89733 | 5.05478 | +0.15745 | 29.725% | 25.999% | -3.726 |
| 4,500 | 4.82437 | 4.84434 | +0.01997 | 30.360% | 27.544% | -2.816 |
| 4,600 | 4.80785 | 4.86476 | +0.05691 | 30.420% | 27.636% | -2.784 |
| 4,700 | 4.82196 | 4.86872 | +0.04676 | 30.621% | 28.623% | -1.998 |
| 4,783 | 4.84920 | 4.94103 | +0.09183 | 29.944% | 27.677% | -2.268 |
| 4,800 | 4.76417 | 4.71558 | -0.04859 | 30.699% | 28.733% | -1.966 |
| 4,900 | 4.79704 | 4.78022 | -0.01682 | 30.616% | 27.860% | -2.757 |
| 5,000 | 4.90505 | 4.95206 | +0.04701 | 30.168% | 25.821% | -4.348 |
| 5,100 | 4.84932 | 4.72705 | -0.12228 | 30.827% | 28.838% | -1.989 |
| 5,200 | 4.95303 | 4.98990 | +0.03687 | 29.501% | 26.218% | -3.282 |
| 5,300 | 4.92026 | 4.86643 | -0.05383 | 30.104% | 26.996% | -3.109 |
| 5,400 | 4.89795 | 5.04650 | +0.14855 | 30.278% | 26.177% | -4.101 |
| 5,500 | 4.97051 | 4.92973 | -0.04078 | 30.731% | 27.311% | -3.420 |
| 5,600 | 4.91201 | 4.77239 | -0.13962 | 31.028% | 28.166% | -2.862 |
| 5,700 | 4.91160 | 4.81773 | -0.09387 | 30.886% | 29.007% | -1.879 |
| 5,800 | 4.85378 | 4.73161 | -0.12218 | 30.927% | 28.902% | -2.025 |
| 5,900 | 4.95112 | 4.77900 | -0.17212 | 30.667% | 28.216% | -2.450 |
| 5,978 | 4.90203 | 4.74913 | -0.15291 | 30.744% | 28.646% | -2.098 |
| 6,000 | 4.94822 | 4.76361 | -0.18461 | 31.279% | 29.149% | -2.130 |
| 6,100 | 4.96095 | 4.73550 | -0.22544 | 31.288% | 29.684% | -1.605 |
| 6,200 | 4.98361 | 4.79584 | -0.18777 | 31.023% | 28.943% | -2.080 |
| 6,300 | 5.07982 | 4.79615 | -0.28367 | 29.999% | 28.129% | -1.870 |
| 6,400 | 5.03585 | 4.75288 | -0.28297 | 30.923% | 29.332% | -1.591 |
| 6,500 | 5.02726 | 4.75662 | -0.27064 | 30.484% | 29.940% | -0.544 |
| 6,600 | 5.01903 | 4.75339 | -0.26565 | 30.900% | 29.359% | -1.541 |
| 6,700 | 5.05077 | 4.77365 | -0.27712 | 30.278% | 29.263% | -1.015 |
| 6,800 | 5.05736 | 4.75468 | -0.30268 | 30.726% | 29.537% | -1.189 |
| 6,900 | 5.02943 | 4.74127 | -0.28816 | 30.968% | 28.468% | -2.501 |
| 7,000 | 5.01249 | 4.76435 | -0.24815 | 31.412% | 29.583% | -1.829 |
| 7,100 | 5.00365 | 4.61843 | -0.38521 | 31.608% | 30.438% | -1.170 |
| 7,173 | 5.01948 | 4.69498 | -0.32450 | 30.703% | 29.135% | -1.568 |
| 7,200 | 4.99681 | 4.60714 | -0.38967 | 31.887% | 30.799% | -1.088 |
| 7,300 | 5.03290 | 4.62874 | -0.40416 | 31.563% | 30.827% | -0.736 |
| 7,400 | 5.03551 | 4.62686 | -0.40865 | 31.942% | 30.612% | -1.330 |
| 7,500 | 5.04466 | 4.68754 | -0.35712 | 32.344% | 29.981% | -2.364 |
| 7,600 | 5.10528 | 4.62020 | -0.48508 | 31.302% | 30.164% | -1.138 |
| 7,700 | 5.08975 | 4.68853 | -0.40122 | 31.453% | 29.793% | -1.660 |
| 7,800 | 5.07503 | 4.65260 | -0.42243 | 31.988% | 31.128% | -0.859 |
| 7,900 | 5.08302 | 4.62613 | -0.45689 | 31.750% | 30.443% | -1.307 |
| 8,000 | 5.09899 | 4.63676 | -0.46223 | 32.235% | 30.319% | -1.916 |
| 8,100 | 5.08298 | 4.56980 | -0.51318 | 31.841% | 29.908% | -1.934 |
| 8,200 | 5.08364 | 4.59928 | -0.48436 | 32.065% | 30.447% | -1.618 |
| 8,300 | 5.06500 | 4.52936 | -0.53564 | 32.262% | 31.476% | -0.786 |
| 8,371 | 5.10064 | 4.55304 | -0.54760 | 31.741% | 31.764% | +0.023 |
| 8,400 | 5.15524 | 4.57852 | -0.57672 | 32.422% | 31.051% | -1.371 |
| 8,500 | 5.16348 | 4.57237 | -0.59111 | 32.116% | 31.773% | -0.343 |
| 8,600 | 5.17643 | 4.65370 | -0.52273 | 32.559% | 31.380% | -1.179 |
| 8,700 | 5.20677 | 4.64060 | -0.56617 | 31.855% | 30.388% | -1.467 |
| 8,800 | 5.19185 | 4.63800 | -0.55386 | 31.311% | 30.177% | -1.134 |
| 8,900 | 5.21708 | 4.57449 | -0.64259 | 31.631% | 31.366% | -0.265 |
| 9,000 | 5.21662 | 4.60418 | -0.61244 | 31.691% | 31.156% | -0.535 |
| 9,100 | 5.22675 | 4.56302 | -0.66372 | 31.928% | 31.270% | -0.658 |
| 9,200 | 5.22696 | 4.57792 | -0.64904 | 31.704% | 31.051% | -0.654 |
| 9,300 | 5.21028 | 4.62416 | -0.58612 | 31.805% | 31.311% | -0.494 |
| 9,400 | 5.20307 | 4.57079 | -0.63229 | 32.280% | 30.909% | -1.371 |
| 9,500 | 5.17181 | 4.53130 | -0.64051 | 32.545% | 31.704% | -0.841 |
| 9,563 | 5.21845 | 4.55647 | -0.66198 | 31.494% | 31.526% | +0.032 |
| 9,600 | 5.18835 | 4.51996 | -0.66840 | 32.417% | 31.727% | -0.690 |
| 9,700 | 5.24425 | 4.51251 | -0.73174 | 32.751% | 31.526% | -1.225 |
| 9,800 | 5.26167 | 4.54387 | -0.71780 | 32.193% | 30.964% | -1.230 |
| 9,900 | 5.28972 | 4.55924 | -0.73048 | 32.513% | 31.348% | -1.166 |
| 10,000 | 5.31044 | 4.57154 | -0.73890 | 32.084% | 31.869% | -0.215 |
| 10,100 | 5.27439 | 4.52067 | -0.75372 | 31.691% | 31.686% | -0.005 |
| 10,200 | 5.33651 | 4.57366 | -0.76285 | 31.558% | 31.211% | -0.347 |
| 10,300 | 5.35438 | 4.57933 | -0.77505 | 32.001% | 31.064% | -0.937 |
| 10,400 | 5.34041 | 4.57025 | -0.77017 | 32.001% | 31.055% | -0.946 |
| 10,500 | 5.31949 | 4.53021 | -0.78928 | 31.933% | 32.084% | +0.151 |
| 10,600 | 5.29604 | 4.52708 | -0.76896 | 32.061% | 31.595% | -0.466 |
| 10,700 | 5.34615 | 4.51733 | -0.82882 | 32.111% | 31.782% | -0.329 |
| 10,766 | 5.35806 | 4.57430 | -0.78376 | 30.927% | 30.150% | -0.777 |
| 10,800 | 5.37728 | 4.55879 | -0.81849 | 32.747% | 31.969% | -0.777 |
| 10,900 | 5.42723 | 4.55406 | -0.87318 | 31.969% | 31.334% | -0.635 |
| 11,000 | 5.39591 | 4.52987 | -0.86603 | 32.276% | 32.020% | -0.256 |
| 11,100 | 5.38663 | 4.51583 | -0.87081 | 32.587% | 32.299% | -0.288 |
| 11,200 | 5.41182 | 4.56590 | -0.84592 | 32.280% | 32.271% | -0.009 |
| 11,300 | 5.43271 | 4.56559 | -0.86712 | 32.312% | 32.344% | +0.032 |
| 11,400 | 5.46842 | 4.57427 | -0.89415 | 31.435% | 31.320% | -0.114 |
| 11,500 | 5.42767 | 4.54461 | -0.88306 | 32.239% | 31.750% | -0.489 |
| 11,600 | 5.43724 | 4.60361 | -0.83363 | 32.253% | 31.915% | -0.338 |
| 11,700 | 5.46002 | 4.56647 | -0.89355 | 32.431% | 32.043% | -0.389 |
| 11,800 | 5.41985 | 4.52159 | -0.89826 | 32.719% | 32.171% | -0.549 |
| 11,900 | 5.41710 | 4.55765 | -0.85945 | 31.672% | 31.764% | +0.091 |
| 11,967 | 5.40964 | 4.54961 | -0.86004 | 31.846% | 31.540% | -0.306 |
| 12,000 | 5.46831 | 4.55472 | -0.91359 | 32.866% | 32.710% | -0.155 |
| 12,100 | 5.47152 | 4.59558 | -0.87594 | 32.614% | 32.289% | -0.325 |
| 12,200 | 5.54698 | 4.63025 | -0.91673 | 32.898% | 32.038% | -0.859 |
| 12,300 | 5.56035 | 4.63816 | -0.92218 | 32.020% | 31.755% | -0.265 |
| 12,400 | 5.47623 | 4.57494 | -0.90130 | 32.198% | 32.344% | +0.146 |

All parameter/graph hashes stayed unchanged, saved validation counts replayed exactly, and native backward-kernel counters remained zero.
Training overlap uses only train rows, casefolded whitespace words and punctuation retained. Short shared phrases alone do not establish memorization. Six fixed greedy samples are descriptive.
