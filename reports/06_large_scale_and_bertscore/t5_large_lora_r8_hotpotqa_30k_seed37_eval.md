# Seq2Seq Upper-Bound Evaluation Report

## Run

- Config: `configs/t5_large_lora_r8_hotpotqa_30k_seed37_eval512.yaml`
- Checkpoint: `runs/t5_large_lora_r8_hotpotqa_30k_seed37/best`
- Data source: `hotpotqa`
- Data objective: `full_reconstruction`
- Seq2Seq model: `google-t5/t5-large`
- Input prefix: `answer question: `

## Metrics

| Metric | Value |
| --- | ---: |
| `validation_loss` | 1.5688 |
| `perplexity` | 4.80 |
| `generation_examples` | 512.0000 |
| `exact_match` | 0.2090 |
| `token_f1` | 0.3273 |
| `source_copy_exact_match` | 0.0000 |
| `source_copy_token_f1` | 0.0104 |
| `prediction_source_token_f1` | 0.0159 |
| `prediction_source_copy_ratio` | 0.7650 |
| `unique_predictions` | 503.0000 |
| `top_prediction_ratio` | 0.0059 |
| `empty_prediction_ratio` | 0.0000 |
| `avg_prediction_length_tokens` | 2.7070 |
| `max_prediction_length_tokens` | 18.0000 |
| `target_len_1_token_f1` | 0.2802 |
| `target_len_1_examples` | 132.0000 |
| `target_len_2_3_token_f1` | 0.3345 |
| `target_len_2_3_examples` | 288.0000 |
| `target_len_4_6_token_f1` | 0.3971 |
| `target_len_4_6_examples` | 79.0000 |
| `target_len_7_plus_token_f1` | 0.2202 |
| `target_len_7_plus_examples` | 13.0000 |
| `token_f1_gain_over_source_copy` | 0.3168 |

## Samples

### Sample 1

- Source: `question: on what street would one find the "journal record building" in oklahoma? context: pierre parrant: pierre "pig's eye" parrant, or pierre parent, was the first person of european descent to live within the borders of what would eventually become the city of saint paul, minnesota. his exploits would eventually propel him to local fame and infamy, in addition to seeing his name briefly adorn the village that would one day become minnesota's capital city. the journal record: the journal record is a daily business and legal newspaper based in oklahoma city, oklahoma. its offices are in downtown oklahoma city, with bureaus at the oklahoma state capitol and in tulsa. josh holliday: josh holliday (born september 14, 1976) is an american college baseball coach and former professional player in minor league baseball. currently the head coach of the oklahoma state cowboys baseball team, he was hired to this position prior to the 2013 season. in 2014, holliday was the big 12 conference baseball coach of the year as osu claimed the conference regular season championship. hollidays' cowboys pulled osu a little cowboy baseball tradition out of the fire and faced oklahoma on the final weekend of 2017. the team was in danger of missing out of the postseason for the 1st time in hollidays tenure at oklahoma state. the cowboys swept the instate rival oklahoma sooners (#2 seed going into region play) to claim the last and final spot as the 8th seed in the bigxii championship. the cowboys went back to their traditionion and won just the 2nd big 12 tournament in schools rich baseball history. the cowboys won`
- Target: `nw 5th`
- Prediction: `downtown oklahoma city, with bureaus at the oklahoma state capitol and in tulsa`
- Exact match: `0.000`
- Token F1: `0.000`
- Prediction-source copy ratio: `0.923`

### Sample 2

- Source: `question: bally's & paris station is on the monorail that is of what length? context: maharajalela monorail station: maharajalela monorail station (previously merdeka station) is a malaysian elevated monorail train station that forms a part of the kuala lumpur monorail (kl monorail) line located in kuala lumpur and opened alongside the rest of the train service on august 31, 2003. tun sambanthan monorail station: tun sambanthan monorail station, formerly sultan sulaiman monorail station, is a malaysian elevated monorail station that forms a part of the kuala lumpur monorail (kl monorail) line located in kuala lumpur and opened alongside the rest of the line and other adjoining monorail stations on august 31, 2003. las vegas monorail: the las vegas monorail is a 3.9 mi monorail mass transit system located adjacent to the las vegas strip, in clark county, nevada, united states. it connects several large casinos in the unincorporated communities of paradise and winchester, and does not enter the city of las vegas. it is owned and operated by the las vegas monorail company. in 2013, total annual ridership was roughly 4.2 million, down from a pre-great recession peak of 7.9 million in 2007. the monorail is a registered not-for-profit corporation, allowed under nevada law since the monorail provides a public service. the state of nevada assisted in bond financing, but no public money was used in construction. bally's &amp; paris station: bally's & paris station is a station on the las vegas monorail. the station is an island platform located at bally's and the paris las vegas hotels. bally's & paris station is located behind the two hotels. medan tuanku monorail`
- Target: `3.9 mi`
- Prediction: `3.9 mi`
- Exact match: `1.000`
- Token F1: `1.000`
- Prediction-source copy ratio: `1.000`

### Sample 3

- Source: `question: name five actors that worked with a german cinematographer? context: fight for fame: fight for fame is a one-hour reality show produced by e! entertainment television, and producers jay james, tim puntillo, alan blassberg, and brian lando. a long established talent agency - acme talent & literary - provides two top agents, adam lieblein (president) and greg meyer to add focus to the show, without being seen as "hosts." each hour shows five actors vying for the opportunity to sign with adam and greg at the agency. they are put through four sets of audition challenges, including monologues, improvisation, and scripted auditions in front of well-known hollywood directors, casting executives and executive producers. at the end of each episode, one actor signs with acme talent & literary. the audience gets to see the decision process of the agents, as well as the attitude of talented and not-so-talented actors. sepp allgeier: josef “sepp” allgeier (6 february 1895 – 11 march 1968) was a german cinematographer who worked on around fifty features, documentaries and short films. he began his career as a cameraman in 1911 for the expreß film co. of freiburg. in 1913 he filmed newsreels in the balkans. he then became an assistant to arnold fanck, a leading director of mountain films. he worked frequently with luis trenker and leni riefenstahl, both closely associated with the genre. he was riefenstahl's lead cameraman on her 1935 propaganda film "triumph of the will". during the second world war, allgeier filmed material for newsreels. he later worked in west german television. his son is the cinematographer hans-jörg allgeier. robert`
- Target: `george clooney, thekla reuten, violante placido, irina björklund, and paolo bonacelli`
- Prediction: `jay james, tim puntillo, alan blassberg, and brian lando`
- Exact match: `0.000`
- Token F1: `0.100`
- Prediction-source copy ratio: `0.889`

### Sample 4

- Source: `question: emu australia is best known for their unisex style boots that are typically made of what? context: cavalier boots: cavalier boots are a style of boot that were popular in europe between approximately 1500-1700 ad. they are soft knee-high leather boots typically made of brown calfskin. tricorne: the tricorne or tricorn is a style of hat that was popular during the 18th century, falling out of style by 1800, though actually not called a "tricorne" until the mid 1800s. during the 18th century hats of this general style were referred to as "cocked hats". at the peak of its popularity, the tricorne varied greatly in style and size, and was worn not only by the aristocracy, but also as common civilian dress, and as part of military and naval uniforms. typically made from animal fiber, the more expensive being of beaver-hair felt and the less expensive of wool felt, the hat's most distinguishing characteristic was that three sides of the brim were turned up (cocked) and either pinned, laced or buttoned in place to form a triangle around the crown. the style served two purposes: first, it allowed stylish gentlemen to show off the most current fashions of their wigs, and thus their social status; and secondly, the cocked hat, with its folded brim, was much smaller than other hats and therefore could be more easily tucked under an arm when going inside a building, where social etiquette dictated that a gentleman remove his hat. tricornes with laced sides could have the laces loosened and the sides dropped down to provide better protection from the weather, sun and rain. boot socks: boot socks`
- Target: `sheepskin`
- Prediction: `brown calfskin`
- Exact match: `0.000`
- Token F1: `0.000`
- Prediction-source copy ratio: `0.500`

### Sample 5

- Source: `question: what film studio produced both national treasure and the computer wore tennis shoes? context: the computer wore tennis shoes: the computer wore tennis shoes is a 1969 american comedy film starring kurt russell, cesar romero, joe flynn and william schallert. it was produced by walt disney productions and distributed by buena vista distribution company as part of "the last laughs of the 1960s". national treasure: book of secrets: national treasure: book of secrets (released on home video as national treasure 2: book of secrets) is a 2007 mystery adventure film directed by jon turteltaub and produced by jerry bruckheimer. it is a sequel to the 2004 film "national treasure" and is the second part of the "national treasure" franchise. the film stars nicolas cage, diane kruger, justin bartha, jon voight, harvey keitel, ed harris, bruce greenwood, and helen mirren. medfield college: medfield college is a fictitious university used as the setting for several films by the walt disney company. among them are the two "professor brainard" movies, "the absent-minded professor" (1961) and "son of flubber" (1963); and the "dexter reilly" trilogy: "the computer wore tennis shoes" (1969), "now you see him, now you don't" (1972), and "the strongest man in the world" (1975); as well as the remake of "the absent-minded professor", "flubber" (1997). sneakers: sneakers (also known as athletic shoes, tennis shoes, gym shoes, runners, takkies, or trainers) are shoes primarily designed for sports or other forms of physical exercise, but which are now also often used for everyday wear. the term generally describes a type of footwear with a flexible sole made of rubber or synthetic material and`
- Target: `walt disney pictures`
- Prediction: `walt disney productions`
- Exact match: `0.000`
- Token F1: `0.667`
- Prediction-source copy ratio: `1.000`

### Sample 6

- Source: `question: the actor who plays the grown version of emma swan's son, from "once upon a time", also plays what character in the television show "the walking dead"? context: andrew j. west: andrew james west is an american character actor who has appeared in films, television series, and shorts. he is best known for his portrayal of fisher in the abc family comedy drama "greek", and more recently as gareth in the amc horror drama "the walking dead". he appeared as a guest star in the fourth-season finale, and was promoted to a series regular for the fifth season. his performance earned him a saturn award for best guest starring role on television nomination. he is also known as playing the older henry mills on abc's "once upon a time". list of once upon a time episodes: "once upon a time" is an american fairy tale drama television series created by edward kitsis and adam horowitz, who also serve as executive producers alongside steve pearlman. it debuted on abc on october 23, 2011. the first season introduces a bail bond agent, emma swan (jennifer morrison) and her birth-son, henry mills (jared s. gilmore), who discover that a new england town named storybrooke, maine is actually a remnant of a parallel world that was cursed by henry's adoptive mother the evil queen/mayor regina mills (lana parrilla) and that all the characters from the fairy tales have no memories of who they were, including the parents of emma: snow white/mary margaret blanchard (ginnifer goodwin) and prince charming/david nolan (josh dallas), who sent her to the real world to save their world and break the curse. carl grimes: carl grimes is a fictional character from the comic book`
- Target: `gareth`
- Prediction: `gareth`
- Exact match: `1.000`
- Token F1: `1.000`
- Prediction-source copy ratio: `1.000`

### Sample 7

- Source: `question: which park is larger of the two, doñana national park or timanfaya national park? context: doñana national park: doñana national park is a natural reserve in andalusia, southern spain, in the provinces of huelva (most of its territory) and seville. it covers 543 km² , of which 135 km² are a protected area. the park is an area of marshes, shallow streams, and sand dunes in las marismas, the delta where the guadalquivir river flows into the atlantic ocean. it was established as a nature reserve in 1969 when the world wildlife fund joined with the spanish government and purchased a section of marshes to protect it. the eco-system has been under constant threat by the draining of the marshes, the use of river water to boost agricultural production by irrigating land along the coast, water pollution by upriver mining, and the expansion of tourist facilities. it is named after wife of the seventh duke of medina-sidonia. yaiza (municipality): yaiza is a municipality on the island of lanzarote in the canary islands, spain. it lies in the southwest of the island and forms part of the province of las palmas. the municipality is bounded by the atlantic ocean to the west, south and east. in the west is the lagoon of charco verde. to the north the timanfaya national park is partly within the municipality. the eastern part of the municipality is mountainous, and south west of the mountains the rubicon plain stretches to the coast. retuerta horse: the retuertas horse, spanish: caballo de las retuertas or caballo de las retuertas de doñana , is a rare breed of horse indigenous`
- Target: `doñana national park`
- Prediction: `timanfaya national park`
- Exact match: `0.000`
- Token F1: `0.667`
- Prediction-source copy ratio: `1.000`

### Sample 8

- Source: `question: which of the actress starred in zombie night appeared in the 1998 erotic thriller "wild things"? context: wild things (film): wild things is a 1998 american erotic thriller film directed by john mcnaughton, and stars matt dillon, neve campbell, kevin bacon, denise richards and theresa russell. wild things: foursome: wild things: foursome is a 2010 erotic thriller film directed by andy hurst and stars jillian murray, marnette patterson, ashley parker angel and john schneider. it is a sequel to "" (2005) and the fourth and final film in the "wild things" series. washington wild things: the washington wild things are a professional baseball team based in the pittsburgh suburb of washington, pennsylvania, in the united states. the wild things are a member of the east division of the frontier league, an independent baseball league which is not affiliated with major league baseball. from the 2002 season to the present, the wild things have played their home games at wild things park. list of erotic thriller films: an erotic thriller is a film genre defined by a thriller with a thematic basis in illicit romance or erotic fantasy. though most erotic thrillers contain scenes of softcore sex, the frequency and explicitness of those scenes varies. if a film is a thriller with scenes of softcore sex or nudity, it is probably not an erotic thriller unless illicit romance or erotic fantasy is central to the dramatic conflict, as in "body heat", "fatal attraction", and "night eyes 3". many crime thrillers, action films, and slasher films contain softcore sex and/or nudity but are not erotic thrillers. likewise, if a film is not identifiably a thriller,`
- Target: `jennifer taylor`
- Prediction: `theresa russell`
- Exact match: `0.000`
- Token F1: `0.000`
- Prediction-source copy ratio: `0.500`
