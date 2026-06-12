# Seq2Seq Upper-Bound Evaluation Report

## Run

- Config: `configs/t5_large_cross_attention_only_squad_30k_seed37.yaml`
- Checkpoint: `runs/t5_large_cross_attention_only_squad_30k_seed37/best`
- Data source: `qa`
- Data objective: `extractive_qa`
- Seq2Seq model: `google-t5/t5-large`
- Input prefix: `answer question: `

## Metrics

| Metric | Value |
| --- | ---: |
| `validation_loss` | 0.3059 |
| `perplexity` | 1.36 |
| `generation_examples` | 256.0000 |
| `exact_match` | 0.6406 |
| `token_f1` | 0.8128 |
| `source_copy_exact_match` | 0.0000 |
| `source_copy_token_f1` | 0.0418 |
| `prediction_source_token_f1` | 0.0414 |
| `prediction_source_copy_ratio` | 0.7917 |
| `unique_predictions` | 255.0000 |
| `top_prediction_ratio` | 0.0078 |
| `empty_prediction_ratio` | 0.0000 |
| `avg_prediction_length_tokens` | 3.0000 |
| `max_prediction_length_tokens` | 14.0000 |
| `target_len_1_token_f1` | 0.8225 |
| `target_len_1_examples` | 78.0000 |
| `target_len_2_3_token_f1` | 0.8314 |
| `target_len_2_3_examples` | 112.0000 |
| `target_len_4_6_token_f1` | 0.8228 |
| `target_len_4_6_examples` | 45.0000 |
| `target_len_7_plus_token_f1` | 0.6559 |
| `target_len_7_plus_examples` | 21.0000 |
| `token_f1_gain_over_source_copy` | 0.7710 |

## Samples

### Sample 1

- Source: `question: what type of vote must the parliament have to either block or suggest changes to the commission's proposals? context: to make new legislation, tfeu article 294 defines the "ordinary legislative procedure" that applies for most eu acts. the essence is there are three readings, starting with a commission proposal, where the parliament must vote by a majority of all meps (not just those present) to block or suggest changes, and the council must vote by qualified majority to approve changes, but by unanimity to block commission amendment. where the different institutions cannot agree at any stage, a "conciliation committee" is convened, representing meps, ministers and the commission to try and get agreement on a joint text: if this works, it will be sent back to the parliament and council to approve by absolute and qualified majority. this means, legislation can be blocked by a majority in parliament, a minority in the council, and a majority in the commission: it is harder to change eu law than stay the same. a different procedure exists for budgets. for "enhanced cooperation" among a sub-set of at least member states, authorisation must be given by the council. member state governments should be informed by the commission at the outset before any proposals start the legislative procedure. the eu as a whole can only act within its power set out in the treaties. teu articles 4 and 5 state that powers remain with the member states unless they have been conferred, although there is a debate about the kompetenz-kompetenz question: who ultimately has the "competence" to define the eu's "competence". many member state courts believe they decide, other member state parliaments believe they decide, while within the eu, the court of justice believes it has the final say.`
- Target: `a majority`
- Prediction: `majority of all meps`
- Exact match: `0.000`
- Token F1: `0.333`
- Prediction-source copy ratio: `1.000`

### Sample 2

- Source: `question: "bairn" and "hyem" have origins from what culture? context: "bairn" and "hyem", meaning "child" and "home", respectively, are examples of geordie words with origins in scandinavia; barn and hjem are the corresponding modern norwegian and danish words. some words used in the geordie dialect are used elsewhere in the northern united kingdom. the words "bonny" (meaning "pretty"), "howay" ("come on"), "stot" ("bounce") and "hadaway" ("go away" or "you're kidding"), all appear to be used in scots; "aye" ("yes") and "nowt" (ipa://naʊt/, rhymes with out,"nothing") are used elsewhere in northern england. many words, however, appear to be used exclusively in newcastle and the surrounding area, such as "canny" (a versatile word meaning "good", "nice" or "very"), "hacky" ("dirty"), "netty" ("toilet"), "hoy" ("throw", from the dutch gooien, via west frisian), "hockle" ("spit").`
- Target: `scandinavia`
- Prediction: `scandinavia`
- Exact match: `1.000`
- Token F1: `1.000`
- Prediction-source copy ratio: `0.000`

### Sample 3

- Source: `question: what did the the europeans think the peoples in the tropics were in need of? context: according to geographic scholars under colonizing empires, the world could be split into climatic zones. these scholars believed that northern europe and the mid-atlantic temperate climate produced a hard-working, moral, and upstanding human being. alternatively, tropical climates yielded lazy attitudes, sexual promiscuity, exotic culture, and moral degeneracy. the people of these climates were believed to be in need of guidance and intervention from the european empire to aid in the governing of a more evolved social structure; they were seen as incapable of such a feat. similarly, orientalism is a view of a people based on their geographical location.`
- Target: `guidance`
- Prediction: `guidance and intervention`
- Exact match: `0.000`
- Token F1: `0.500`
- Prediction-source copy ratio: `1.000`

### Sample 4

- Source: `question: what do students learn about in confirmation and membership preparation classes? context: unlike confirmation and profession of faith, baptism is a sacrament in the umc. the book of discipline of the united methodist church directs the local church to offer membership preparation or confirmation classes to all people, including adults. the term confirmation is generally reserved for youth, while some variation on membership class is generally used for adults wishing to join the church. the book of discipline normally allows any youth at least completing sixth grade to participate, although the pastor has discretionary authority to allow a younger person to participate. in confirmation and membership preparation classes, students learn about church and the methodist-christian theological tradition in order to profess their ultimate faith in christ.`
- Target: `church and the methodist-christian theological tradition`
- Prediction: `church and the methodist-christian theological tradition`
- Exact match: `1.000`
- Token F1: `1.000`
- Prediction-source copy ratio: `1.000`

### Sample 5

- Source: `question: what are the two major thoroughfares of sunnyside? context: the neighborhood of sunnyside is on fresno's far southeast side, bounded by chestnut avenue to the west. its major thoroughfares are kings canyon avenue and clovis avenue. although parts of sunnyside are within the city of fresno, much of the neighborhood is a "county island" within fresno county. largely developed in the 1950s through the 1970s, it has recently experienced a surge in new home construction. it is also the home of the sunnyside country club, which maintains a golf course designed by william p. bell.`
- Target: `kings canyon avenue and clovis avenue`
- Prediction: `kings canyon avenue and clovis avenue`
- Exact match: `1.000`
- Token F1: `1.000`
- Prediction-source copy ratio: `1.000`

### Sample 6

- Source: `question: what does the zachęta national gallery of art organize exhibitions of art from? context: the 17th century royal ujazdów castle currently houses centre for contemporary art, with some permanent and temporary exhibitions, concerts, shows and creative workshops. the centre currently realizes about 500 projects a year. zachęta national gallery of art, the oldest exhibition site in warsaw, with a tradition stretching back to the mid-19th century organises exhibitions of modern art by polish and international artists and promotes art in many other ways. since 2011 warsaw gallery weekend is held on last weekend of september.`
- Target: `polish and international artists`
- Prediction: `modern art by polish and international artists`
- Exact match: `0.000`
- Token F1: `0.727`
- Prediction-source copy ratio: `1.000`

### Sample 7

- Source: `question: what position did luther have in wittenberg? context: on 19 october 1512, he was awarded his doctor of theology and, on 21 october 1512, was received into the senate of the theological faculty of the university of wittenberg, having been called to the position of doctor in bible. he spent the rest of his career in this position at the university of wittenberg.`
- Target: `doctor in bible`
- Prediction: `doctor in bible`
- Exact match: `1.000`
- Token F1: `1.000`
- Prediction-source copy ratio: `0.667`

### Sample 8

- Source: `question: who kicked ethelred out? context: the normans were in contact with england from an early date. not only were their original viking brethren still ravaging the english coasts, they occupied most of the important ports opposite england across the english channel. this relationship eventually produced closer ties of blood through the marriage of emma, sister of duke richard ii of normandy, and king ethelred ii of england. because of this, ethelred fled to normandy in 1013, when he was forced from his kingdom by sweyn forkbeard. his stay in normandy (until 1016) influenced him and his sons by emma, who stayed in normandy after cnut the great's conquest of the isle.`
- Target: `sweyn forkbeard`
- Prediction: `sweyn forkbeard`
- Exact match: `1.000`
- Token F1: `1.000`
- Prediction-source copy ratio: `0.500`
