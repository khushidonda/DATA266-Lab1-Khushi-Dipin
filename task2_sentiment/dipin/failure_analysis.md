# Task 2: Error Review (Dipin)

Model reviewed: **experimental_2** (2-layer BiLSTM + additive attention; best validation macro-F1 = 0.9508)
Source: `outputs/error_review/error_candidates_experimental_2.csv` (official test set; full review text is in the CSV)
Labels: 0 = negative, 1 = positive. P(pos) = model's predicted probability of positive. "Top attention" = the 5 processed tokens with the highest attention weight in experimental_2.

Error-type vocabulary (used below): <!-- e.g. define your own labels, such as sarcasm, mixed sentiment, negation scope, label noise, domain-specific term, truncation, OOV-heavy -->

## A. Confident false positives
_true negative, predicted positive with high confidence_

| # | Test idx | P(pos) | Snippet | Top attention | Error type | Observation |
|---|---|---|---|---|---|---|
| 1 | 29330 | 1.0000 | "Wow love the place and everything is very clean and new! Great place to come and relax worth a try! Cheers, Eric Van Nguyen Visited April 2012" | love, great, worth, clean, relax | | |
| 2 | 29923 | 0.9999 | "Pumpkin Spice. Red Velvet Cake. Friggin NUTELLA. These are the flavors I love the most...if any other froyo joint made them. I have passed by this place many times…" | favorite, nutella, love, nutella, enjoy | | |
| 3 | 5185 | 0.9995 | "Like Clay P who posted before me, I too love pancakes. Though I love chocolate chip pancakes. But like Clay I did not love the ones that I got a the Original Pancake House…" | definitely, respectable, job, clay, worthwhile | | |
| 4 | 15988 | 0.9993 | "Local dining icons deserve their own criteria for recognition. That said, adding sliced tomato, coleslaw and a pile of fries to a deli sandwich defines the Primanti as a Pittsburgh-area institution. Their cheesesteaks will never threaten the best of Philly's…" | enjoyable, imperfection, always, adventure, best | | |
| 5 | 4488 | 0.9993 | "It's good. The rolls are better than the sashimi although one time we had some really nice (and surprising) maguro. We always get the hamachi jalapeño roll and some sashimi so don't get me wrong…" | enjoyable, good, importantly, good, nice | | |

## B. Confident false negatives
_true positive, predicted negative with high confidence_

| # | Test idx | P(pos) | Snippet | Top attention | Error type | Observation |
|---|---|---|---|---|---|---|
| 6 | 22807 | 0.0001 | "EDIT: They really did change the service up since I last posted this. Horrible service. Used to be my favorite pizza in the city (at a reasonable price), but I'm rethinking that. We just had an altercation with a server…" | disrespect, not, favorite, horrible, party | | |
| 7 | 11401 | 0.0004 | "Perhaps my expectations were too high because of all the hype I'd heard, but I have to say I was a little disappointed. We did a girls' night out… after waiting forever for a table…" | hype, disappoint, expectation, high, cold | | |
| 8 | 25329 | 0.0004 | "Im with Kimberly B...this is a card for people rebuilding credit. Mine got tanked due to a divorce. I have been rebuilding, and started with a $300 limit, and it is now at $1250. I do not carry a balance, ever…" | 1250, blame, not, disclose, not | | |
| 9 | 30793 | 0.0004 | "This place is so much better since they changed owners. My wife and I went when it was the old owners, it was terrible. We waited forever and the food never came before we walked out…" | terrible, horrible, never, food, forever | | |
| 10 | 12280 | 0.0007 | "Came here back in June for the Britney Spears concert with my daughter. Not the best accoustics, but still a fairly nice venue. This was the most bizarre concert I've ever attended. It's not the venue's fault…" | poor, bewilder, misfortune, f, verbally | | |

## C. Near-threshold errors
_P(positive) close to 0.5_

| # | Test idx | P(pos) | Snippet | Top attention | Error type | Observation |
|---|---|---|---|---|---|---|
| 11 | 4169 | 0.5005 (true neg) | "Well it's over when they want it to be underwear night 11-to whenever they feel it's over" | underwear, well, whenever, night, want | | |
| 12 | 1772 | 0.5005 (true neg) | "Definitely not one of the nicer Hiltons I've stayed at, but close to the airport...so its convenient and the inside of the place is nice enough. Not the place you want to leave ANY valuables in your car…" | definitely, not, cheeper, not, indication | | |
| 13 | 29447 | 0.4993 (true pos) | "Great place to feed your caffeine addiction. Quality coffee, lattes, and tea. The menu is limited. I ordered the chicken salad. Not bad at all, but the sandwich was thrown together and the chips were from the bottom of the bag…" | great, stale, throw, place, least | | |
| 14 | 8723 | 0.4991 (true pos) | "went in tonight on my way home and scored two pairs of $35 sweats for $12 each, yea!!! also found a beanie for my kid for $7 not such a great deal but it does have a Nike swoosh on it…" | score, miss, not, two, pair | | |
| 15 | 5190 | 0.5017 (true neg) | "After shopping on-and-off here for nearly 5-years, it's time for a quick update, and a slight downgrade. - Parking lot. Always cramped, especially when they park a 53' long refrigerated semi trailer…" | 53, knowledgeable, sinful, wrinkly, shrivel | | |

## D. Slice-specific failures
_worst slice: **high_unk_rate (> 10% of tokens are `<UNK>`)**, n = 231, error rate 14.72% for experimental_2 (vs 4.67% overall). All 5 below are true negative, predicted positive._

| # | Test idx | P(pos) | Snippet | Top attention | Error type | Observation |
|---|---|---|---|---|---|---|
| 16 | 1256 | 0.9990 | "What era is this? That was my first thought as I stepped into this restaurant. Mirrored ceilings, velvet upholstered chairs, cheetah print fabric -- it was like I was in the 80's or in the movie "Scarface."…" | delicious, woodsy, escargot, cheetah, escargot | | |
| 17 | 36104 | 0.9985 | "Das HubRaum ist für mich die Sommerkneipe in Durlach. Schön ruhig und im Grünen gelegen, bietet es reichlich Parkplätze, die aber auch belegt sind wenn das Lokal voll wird…" (German) | bietet, prima, reichlich, al, die | | |
| 18 | 35963 | 0.9968 | "Interessante Gestaltung, mitten in der Stadt, unfreundliche Bedienung und ekelhafte Getränke, ein Stern ist da fast zu viel." (German) | fast, ist, zu, viel, u00e4nke | | |
| 19 | 19328 | 0.9952 | "On n'y va QUE pour la section fruits et légumes, toujours bien remplie. Sinon, c'est sale! Les étagères sont sans dessus dessous, à moitié vides…" (French) | bien, remplie, toujours, u00e9gumes, dessous | | |
| 20 | 27150 | 0.9923 | "Die Cocktails waren einen Besuch wert, alles andere war eher enttäuschend bei unserem Besuch. Ideal zentral und ein wenig ausgefallen (Prinz-Max-Palais) gelegen…" (German) | palais, wert, freundliche, ideal, entt | | |

## Patterns across the 20 errors
<!-- your own analysis -->

## Proposed testable fix
- Fix:
- Hypothesis:
- How to test it (metric, slice, expected change):
