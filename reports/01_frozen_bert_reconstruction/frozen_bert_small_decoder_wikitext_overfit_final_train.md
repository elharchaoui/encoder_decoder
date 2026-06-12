# Experiment Evaluation Report

## Run

- Config: `configs/frozen_bert_small_decoder_wikitext_overfit.yaml`
- Checkpoint: `runs/frozen_bert_small_decoder_wikitext_overfit/final.pt`
- Data source: `wikitext`
- Encoder: `google-bert/bert-base-uncased`
- Decoder layers: `4`
- Decoder heads: `8`
- Encoder frozen: `true`
- Decoder embedding init from encoder: `True`
- Token embeddings tied: `True`

## Metrics

| Metric | Value |
| --- | ---: |
| `train_loss` | 0.0042 |
| `perplexity` | 1.00 |
| `generation_examples` | 64.0000 |
| `exact_match` | 0.0000 |
| `token_f1` | 0.5411 |
| `source_copy_exact_match` | 0.0000 |
| `source_copy_token_f1` | 0.7867 |
| `encoder_calls_per_generation` | 1.0000 |
| `token_f1_gain_over_source_copy` | -0.2456 |

## Samples

### Sample 1

- Source: `[MASK] player character is ardan  an eccentric and [MASK] scientist who is [MASK]  daring and cheerful president barbicane [MASK] the president [MASK] the [MASK] club and captain nicholl   are both found dead the start game  not having survived flight the moon  a woman called diana features in game 's backstory  as a woman whose ancestors made contact the selenites  [MASK] these [MASK] characters  there are also several selenite characters such [MASK] supreme moon ruler  the high dignitary [MASK]  scruple  and the three exiles `
- Target: `the player character is michel ardan , an eccentric and intrepid french scientist who is enthusiastic , daring and cheerful . president barbicane , the president of the gun club , and captain nicholl , an engineer , are both found dead at the start of the game , not having survived the flight to the moon . a woman called diana features in the game 's backstory , as a woman whose ancestors made contact with the selenites . apart from these human characters , there are also several selenite characters such as the supreme moon ruler , the high dignitary , scurvy , scruple , and the three exiles .`
- Prediction: `the player character is michel ardan, an eccentric and intrepid french scientist who is enthusiastic, daring and cheerful. president barbicane, the president of the gun club, and captain nicholl, an engineer, are both found dead at the start of the game, not having survived the flight to`
- Exact match: `0.000`
- Token F1: `0.491`

### Sample 2

- Source: `regardless of the titles odaenathus controlled roman east with approval of who could do little but formalize odaenathus self achieved status and [MASK] his [MASK] loyalty  palmyra  although officially still of the roman empire  became a de facto state to rome instead of a provincial city  outside palmyra  odaenathus ' authority extended from pontic coast in the north in the south  this area included roman provinces of syria   palaestina  arabia  anatolia [MASK] eastern regions later following the campaign ) osroene and mesopotamia `
- Target: `regardless of the titles , odaenathus controlled the roman east with the approval of gallienus who could do little but to formalize odaenathus self achieved status and settle for his formal loyalty . palmyra itself , although officially still part of the roman empire , became a de facto allied state to rome instead of a provincial city . outside of palmyra , odaenathus ' authority extended from the pontic coast in the north to palestine in the south . this area included the roman provinces of syria , phoenice , palaestina , arabia , anatolia 's eastern regions and later ( following the campaign of 262 ) osroene and mesopotamia .`
- Prediction: `regardless of the titles, odaenathus controlled the roman east with the approval of gallienus who could do little but to formalize odaenathus self achieved status and settle for his formal loyalty. palmyra itself, although officially still part of the roman empire, became a de facto allied state to`
- Exact match: `0.000`
- Token F1: `0.550`

### Sample 3

- Source: `ímar 's death  diarmait appears have appointed his own son  murchad ( died [MASK] )  control of dublin later that decade  as annals the four masters accords him the title tigherna gall  meaning " lord the " 1059  [MASK] 1061  murchad invaded mann and seems to have overthrown echmarcach  both [MASK] son were dead by 1072  and the annals tigernach describes diarmait on his death that year as king [MASK] the [MASK] ( rí [MASK] gall  literally " king of the isles of the [MASK] "  declaration which seems to indicate [MASK]  by [MASK] century at  the kingship of the isles was [MASK] upon control mann `
- Target: `after ímar 's death , diarmait appears to have appointed his own son , murchad ( died 1070 ) , control of dublin later that decade , as the annals of the four masters accords him the title tigherna gall , meaning " lord of the foreigners " in 1059 . in 1061 , murchad invaded mann and seems to have overthrown echmarcach . both father and son were dead by 1072 , and the annals of tigernach describes diarmait on his death that year as king of the isles ( rí innsi gall , literally " king of the isles of the foreigners " ) , a declaration which seems to indicate that , by the eleventh century at least , the kingship of the isles was contingent upon control of mann .`
- Prediction: `after imar ' s death, diarmait appears to have appointed his own son, murchad ( died 1070 ), control of dublin later that decade, as the annals of the four masters accords him the title tigherna gall, meaning " lord of the foreigners " in 1059`
- Exact match: `0.000`
- Token F1: `0.422`

### Sample 4

- Source: `[MASK] nine electors were created clement xiii  while fifteen by benedict  alessandro albani received the red [MASK] innocent xiii  and neri maria corsini from clement xii `
- Target: `twenty nine electors were created by clement xiii , while fifteen by pope benedict xiv . alessandro albani received the red hat from innocent xiii , and neri maria corsini from clement xii .`
- Prediction: `twenty nine electors were created by clement xiii, while fifteen by pope benedict xiv. alessandro albani received the red hat from innocent xiii, and neri maria corsini from clement xii.`
- Exact match: `0.000`
- Token F1: `0.812`

### Sample 5

- Source: `my divisional generals and brigadiers [MASK] to me that they fear their [MASK] are thoroughly demoralised by shrapnel fire which they have been subjected all day after exhaustion and gallant work morning  numbers have dribbled back from the firing line and cannot be collected in this country  even new zealand brigade which has only recently been engaged lost and is to some demoralised  if [MASK] are subjected to shellfire again tomorrow morning there is likely to be [MASK] fiasco [MASK] as i have no fresh troops which to those in firing line  [MASK] know my representation is most serious  [MASK] [MASK] we are [MASK] @-@ it must be once [MASK]`
- Target: `both my divisional generals and brigadiers have represented to me that they fear their men are thoroughly demoralised by shrapnel fire to which they have been subjected all day after exhaustion and gallant work in morning . numbers have dribbled back from the firing line and cannot be collected in this difficult country . even new zealand brigade which has only recently been engaged lost heavily and is to some extent demoralised . if troops are subjected to shellfire again tomorrow morning there is likely to be a fiasco , as i have no fresh troops with which to replace those in firing line . i know my representation is most serious , but if we are to re @-@ embark it must be at once .`
- Prediction: `both my divisional generals and brigadiers have represented to me that they fear their men are thoroughly demoralised by shrapnel fire to which they have been subjected all day after exhaustion and gallant work in morning. numbers have dribbled back from the firing line and cannot be collected in this difficult country.`
- Exact match: `0.000`
- Token F1: `0.559`
