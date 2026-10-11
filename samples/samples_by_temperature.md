# Qualitative Text Generation Samples across Models and Temperatures

**Dataset**: The Time Machine by H. G. Wells
**Prompt Seeds**: 4 diverse seed sentences
**Sampling Temperatures**: [0.2, 0.7, 1.2] (Low: 0.2, Moderate: 0.7, High: 1.2)

## Seed Prompt: *"the time traveller began to explain the strange"*

### Simple RNN

| Temperature | Generated Continuation | Distinct-1 | Distinct-2 | Rep-3 (%) |
|:---:|:---|:---:|:---:|:---:|
| 0.2 | incredible thing the world will go then i saw a curved view of the world was only a brilliant who had seen the sound of the family and the only tears except my own hint | 0.7714 | 0.9412 | 0.0% |
| 0.7 | flowers i determined to be in this affection was absolutely wrong and i felt that in strange further people upon the morlocks was better hands me i felt that was terror of the sun was | 0.7429 | 0.9412 | 3.0% |
| 1.2 | slope you but some across air what no mere wood of the best tangle of an circular roof of a certain tangle of paper between at the pedestal of shoes to account one he touching | 0.8286 | 0.9706 | 0.0% |

### Bidirectional RNN

| Temperature | Generated Continuation | Distinct-1 | Distinct-2 | Rep-3 (%) |
|:---:|:---|:---:|:---:|:---:|
| 0.2 | white flowers preserved i fancied i saw our own time machine then i tried to the sun was a sense of fear grew over its long or so better of them a sound of my | 0.8571 | 1.0 | 0.0% |
| 0.7 | laughter of the palace of green porcelain as i was not been he said i travelled into my truth to you all going to the morlocks to the smoking room as i travelled into a | 0.7143 | 0.8824 | 3.0% |
| 1.2 | time machine so i had a queer dimension of his voyage time they stayed loose part of the intellectual beautiful of that results in eight dimensions stone next life in the rising moon weena grew | 0.8571 | 1.0 | 0.0% |

### LSTM

| Temperature | Generated Continuation | Distinct-1 | Distinct-2 | Rep-3 (%) |
|:---:|:---|:---:|:---:|:---:|
| 0.2 | thing filby easier at right angles to the time traveller and the lamp and the lamp was very hot thing i saw the time traveller proceeded the psychologist was generally complete but you cannot move | 0.7143 | 0.8824 | 6.1% |
| 0.7 | thing looking round me for a minute he walked over the machine he must have been him is a fourth dimension of his cigar a trick waned of cheerfulness he he said the lamp and | 0.8 | 1.0 | 0.0% |
| 1.2 | flowers sitting in the summit of the certain travelling worn too too again touched me to break speed i had most of ruins i had seen forthwith to open centuries but these other not followed | 0.8286 | 0.9706 | 0.0% |

### GRU

| Temperature | Generated Continuation | Distinct-1 | Distinct-2 | Rep-3 (%) |
|:---:|:---|:---:|:---:|:---:|
| 0.2 | thing he walked slowly and the pale blue champagne of the absence of his gesture and looking round the psychologist s a certain weakness at one who had come to a touch of meat i | 0.8286 | 1.0 | 0.0% |
| 0.7 | thing his coat was a stir of thunder for a time machine i was just to myself that it meant somehow to me in the only there was the same hall with dark in the | 0.8 | 0.9706 | 0.0% |
| 1.2 | way this they seemed at the laboratory you but you what said filby s psychologist was almost recovered from the lamp that the fact is the dream so he went out his eye room the | 0.8571 | 1.0 | 0.0% |

## Seed Prompt: *"it is simply that our consciousness moves along it"*

### Simple RNN

| Temperature | Generated Continuation | Distinct-1 | Distinct-2 | Rep-3 (%) |
|:---:|:---|:---:|:---:|:---:|
| 0.2 | as i have been a queer thing as a ghost of the morlocks i had already shown him without further round to the morlocks and the eloi were splendid and that i had followed myself | 0.7429 | 0.9412 | 0.0% |
| 0.7 | as i did so the full of my sleep i will believe that is him to the machine it s plain that he was mayor like a fire as a thud perhaps a mile across | 0.8286 | 1.0 | 0.0% |
| 1.2 | as you have been seated say as of bronze doors the floor was lit a sleep they merged at the thing i suppose i move about nevertheless a question did not trouble good past them | 0.8857 | 1.0 | 0.0% |

### Bidirectional RNN

| Temperature | Generated Continuation | Distinct-1 | Distinct-2 | Rep-3 (%) |
|:---:|:---|:---:|:---:|:---:|
| 0.2 | was full great our fear was i determined to the sun was rather changed and the floor of a long i could not fall back for a little eyes or less next hand and every | 0.8286 | 1.0 | 0.0% |
| 0.7 | was full great green structure for these an explanation very clear in his cupola or the white man whom there was still over to get that i was too of an explosion for almost unfamiliar | 0.8857 | 1.0 | 0.0% |
| 1.2 | in knowledge of my feet that surrounded for nine in my hand and indeed i was seeking sleeping across the other great thunder of a men of self red end than since itself to come | 0.8857 | 1.0 | 0.0% |

### LSTM

| Temperature | Generated Continuation | Distinct-1 | Distinct-2 | Rep-3 (%) |
|:---:|:---|:---:|:---:|:---:|
| 0.2 | isn t the time traveller smiled in his hand and the lamp he said he said the psychologist will conclude was expounding a misconception is a minute is a certain faltering articulation i m a | 0.7714 | 0.9412 | 0.0% |
| 0.7 | well as long thursday i felt a very young man had followed us the time traveller seemed to his hand and we could not he say i get this way marking the geometry of a | 0.9143 | 1.0 | 0.0% |
| 1.2 | night and a red splendid little weena occupied not the difference of fear are watching even the metallic things so cheerfully even sticks and when the silver had been great form to judge as the | 0.8571 | 1.0 | 0.0% |

### GRU

| Temperature | Generated Continuation | Distinct-1 | Distinct-2 | Rep-3 (%) |
|:---:|:---|:---:|:---:|:---:|
| 0.2 | can you you you will you convince me at first time for this fourth dimension it was not only the fact of the morning they had been devoted to a little people under the future | 0.8571 | 0.9706 | 0.0% |
| 0.7 | may be be out of us to judge by which there is another at first but so for such a kind of the morlocks so that i determined to come for my watch it were | 0.8571 | 1.0 | 0.0% |
| 1.2 | took us at shadows became a rhododendron beautiful quiet eight ever and graceful darkness for that pace largely seemed odd to rise to the machine and of sticks and sufficient apparatus out of self life | 0.8857 | 1.0 | 0.0% |

## Seed Prompt: *"i looked at the small machine and wondered how"*

### Simple RNN

| Temperature | Generated Continuation | Distinct-1 | Distinct-2 | Rep-3 (%) |
|:---:|:---|:---:|:---:|:---:|
| 0.2 | to think that is all that had been almost set me left fingers and fell over the same thud thud the morlocks so a mere paradox of our own time i was almost call the | 0.8571 | 1.0 | 0.0% |
| 0.7 | the medical man has make myself about the morlocks grey better gesture and a natural lawn in the same way i was no feeling as i walked lit a new laughing complex little they were | 0.8857 | 1.0 | 0.0% |
| 1.2 | raced as swiftly as it might be realise my confident folly of below it the machine would tumult i old world since that was their lack of human he along its long was as if | 0.8571 | 1.0 | 0.0% |

### Bidirectional RNN

| Temperature | Generated Continuation | Distinct-1 | Distinct-2 | Rep-3 (%) |
|:---:|:---|:---:|:---:|:---:|
| 0.2 | to myself that i lay her to be that from what had been no small all to strike before i had happened and the thought of my fist i gave a little problem of the | 0.7714 | 1.0 | 0.0% |
| 0.7 | i was clear abundant towards the little of disease filby proved in one has brilliant the larger than an all garden followed of them among these golden child we were clear enough to the same | 0.8857 | 1.0 | 0.0% |
| 1.2 | examination of terror in the one of white emerged from examining me and for this fourth i wondered reputations as i had my hand on the smoking room was incredible apparently their way upon some | 0.9143 | 1.0 | 0.0% |

### LSTM

| Temperature | Generated Continuation | Distinct-1 | Distinct-2 | Rep-3 (%) |
|:---:|:---|:---:|:---:|:---:|
| 0.2 | to the same beautiful and the morlocks had been attained of the same splendid air of the morlocks and the morlocks and the same splendid palaces of the morlocks had been attained that the morlocks | 0.4 | 0.5588 | 24.2% |
| 0.7 | to the underworld was almost immediately came straight towards the end and the end of a minute he was not a friend of the machine had been unreasonable i had seen a mile and shoulders | 0.7143 | 0.9706 | 0.0% |
| 1.2 | for suit the morlocks get it without for a hands or laughing convey my sandals were now with fire and struck me such several larger the clock rose into my generation which i could once | 0.9143 | 1.0 | 0.0% |

### GRU

| Temperature | Generated Continuation | Distinct-1 | Distinct-2 | Rep-3 (%) |
|:---:|:---|:---:|:---:|:---:|
| 0.2 | to me and the morlocks and the morlocks to mere narrow footprints my general impression of it and they had seen none i took up the morlocks and the morlocks and the morlocks and the | 0.5714 | 0.6765 | 27.3% |
| 0.7 | to me i was surprised to come with a long vigil or they were not the truth i determined to do not telling myself but came up and dragging up a round the white sphinx | 0.8 | 1.0 | 0.0% |
| 1.2 | none stayed my hand lit an truth it seemed to me that the darkness turn when she opened from her curiously grotesque loose and as i struggled with sleep when she was not before her | 0.9143 | 0.9706 | 0.0% |

## Seed Prompt: *"the world was completely changed in the distant future"*

### Simple RNN

| Temperature | Generated Continuation | Distinct-1 | Distinct-2 | Rep-3 (%) |
|:---:|:---|:---:|:---:|:---:|
| 0.2 | were soon was a deep thing as i had seen near the floor and so evidently i had struggled down the same thud thud the morlocks i could imagine the machine and the match and | 0.7143 | 0.9706 | 0.0% |
| 0.7 | were following me to look at the sides of the roof was lost of dust and sliding out of the morlocks i suppose i do not better confess but weena came tired a little while | 0.8571 | 0.9706 | 0.0% |
| 1.2 | were simply it could gives you set up and hands hesitating in dust and like door touched that perfect they had been a line of yards on the eternal sky in my hand about the | 0.9143 | 1.0 | 0.0% |

### Bidirectional RNN

| Temperature | Generated Continuation | Distinct-1 | Distinct-2 | Rep-3 (%) |
|:---:|:---|:---:|:---:|:---:|
| 0.2 | were sleeping they were very feeling exactly the morlocks and i must have been free i felt my own circumstances over to the morlocks to be the time traveller stooping to the first i had | 0.7429 | 0.9412 | 0.0% |
| 0.7 | were here except him and still move for it was a thing i felt appetite you really explain if you may understand the distant specimens of wait and i determined to what well it s | 0.8857 | 1.0 | 0.0% |
| 1.2 | home to learn some beating of granite stone and co the sun followed breaking came upon and merely splendid only the night of the long tunnel of men this thought of my difficulties i went | 0.8286 | 1.0 | 0.0% |

### LSTM

| Temperature | Generated Continuation | Distinct-1 | Distinct-2 | Rep-3 (%) |
|:---:|:---|:---:|:---:|:---:|
| 0.2 | of our own part i had seen the same rich had been devoted to be explicit i had seen them as i was sorry to me that the morlocks had been glad to mount but | 0.7143 | 0.9118 | 3.0% |
| 0.7 | were illuminated i saw it seemed to me was very much to the left of physical moment he was all the lever at my hand here suddenly i saw a long great way under the | 0.8286 | 0.9706 | 0.0% |
| 1.2 | scarcely other about no absolute folly of horrible strong i looked faded were slipping and tall night and fell about gallery were common fruits as i could weena far a great perhaps through that it | 0.8857 | 1.0 | 0.0% |

### GRU

| Temperature | Generated Continuation | Distinct-1 | Distinct-2 | Rep-3 (%) |
|:---:|:---|:---:|:---:|:---:|
| 0.2 | were the same red old lesson of the land in the air that the same triumph of my arrival i fancied i was in my hands i felt a match but at first i had | 0.7143 | 0.9706 | 0.0% |
| 0.7 | of being force and then when i ve overlooked matches the morlocks as i set my iron crowbar and that morning my interpretation with it in my right the corner of weena or three feet | 0.8286 | 1.0 | 0.0% |
| 1.2 | of cattle a brilliant more serious in profound yet rather engaged and the received there was for the too say itself in that horizontal world beach below saw this foliage masonry it was not only | 0.9143 | 1.0 | 0.0% |

