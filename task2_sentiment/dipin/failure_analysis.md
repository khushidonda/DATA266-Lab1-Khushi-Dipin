# Task 2: Error Review (Dipin)

**Model reviewed:** experimental_2 (2-layer BiLSTM + attention). It is the best of my three models on validation (macro-F1 0.9508).
It gets 1,774 of the 38,000 test reviews wrong (4.67%). This file looks at 20 of them.

**How the 20 were picked:** automatically by the notebook, not by hand. It takes the 5 most confident false positives, the 5 most confident false negatives, the 5 errors closest to P = 0.5, and the 5 most confident errors in the worst data slice.
The full review texts are in `outputs/error_review/error_candidates_experimental_2.csv`.

**Reading the tables:** label 0 = negative, 1 = positive. P(pos) is the model's probability that the review is positive. "The model saw" means the text after my cleaning step (lowercased, punctuation and stopwords removed, lemmatized).

## Error types used

| Error type | What it means | Count |
|---|---|---|
| Mixed review | Real praise and real complaints in the same review. The label depends on which side the writer weighed more. | 7 |
| Unknown words | The review is not in English, or is full of tokens the model never learned (links, broken accents). | 5 |
| Sentiment about something else | The strong words are about the past, other customers or other reviewers, not about the business as it is now. | 3 |
| Label noise | The text clearly says one thing and the label says the other. The model's answer is reasonable. | 2 |
| Indirect negative | Positive words sit inside a negative comparison or next to a low score. | 2 |
| Too little signal | Very short or neutral text with no clear opinion. | 1 |

## A. Confident false positives
_True label negative, model very sure it is positive._

| # | Test idx | P(pos) | Snippet | Error type | What went wrong |
|---|---|---|---|---|---|
| 1 | 29330 | 1.0000 | "Wow love the place and everything is very clean and new! Great place to come and relax worth a try!" | Label noise | Nothing in the text is negative. The model attends to *love, great, worth, clean, relax* and says positive, which matches the text. The label looks wrong. |
| 2 | 29923 | 0.9999 | "These are the flavors I love the most...if any other froyo joint made them. … I would recommend just about every other froyo joint than this one." | Indirect negative | The writer uses *love*, *favorite* and *recommend*, but about other shops. My stopword list deletes *other*, *than* and *this*, so the model saw "would recommend every froyo joint one", which reads as praise. |
| 3 | 5185 | 0.9995 | "But like Clay I did not love the ones that I got … they did a respectable job. It is definitely a worthwhile destination for a pancake lover." | Mixed review | One complaint early, then clear praise at the end. The model follows the ending (*respectable, definitely, worthwhile*). The negative label is debatable. |
| 4 | 15988 | 0.9993 | "Personally, I'm not a fan of loading a sandwich with fries … the conclusion I've always had was of an enjoyable adventure. Prices are downright cheap." | Mixed review | The criticism is polite and indirect ("not a fan", "an acquired taste"). The praise is direct ("generally good", "enjoyable", "cheap"). The model counts the direct words. The label is debatable. |
| 5 | 4488 | 0.9993 | "It's good. … It's still enjoyable. … Little Tokyo in Mt. Lebanon and Kiku in Station Square are better. … 4.5 of 10 relative to sushi quality alone" | Indirect negative | Almost every opinion word is positive (*good, enjoyable, nice*). The real verdict is a score and a comparison. "4.5 of 10" becomes the tokens "4 5 10", which mean nothing to the model. |

## B. Confident false negatives
_True label positive, model very sure it is negative._

| # | Test idx | P(pos) | Snippet | Error type | What went wrong |
|---|---|---|---|---|---|
| 6 | 22807 | 0.0001 | "Horrible service. Used to be my favorite pizza in the city (at a reasonable price), but I'm rethinking that." | Label noise | The whole text is a complaint about a rude server. The review starts with "EDIT:", so the text may have been rewritten after the rating was given. The model's answer matches the text. |
| 7 | 11401 | 0.0004 | "I have to say I was a little disappointed. … My suggestion is head there for drinks but not for food." | Mixed review | Most of the text is complaints (long wait, wrong food, cold food). The writer still rated it positive because of the drinks. The model attends to *hype, disappoint, cold*. The label is debatable. |
| 8 | 25329 | 0.0004 | "Also, it is NOT A HIDDEN \$75 fee. It isn't hidden! … Stop blaming the company for being a business and take your finances into your own hands." | Sentiment about something else | The writer defends a credit card and is angry at other reviewers. Words like *blame*, *late*, *divorce* and *fee* sound negative, but they are not aimed at the business. |
| 9 | 30793 | 0.0004 | "This place is so much better since they changed owners. … It was horrible. Now its much better. … I have nothing but positive things to now say about this place." | Sentiment about something else | The bad words describe the old owners. The model attends to *terrible, horrible, never, forever*. My stopword list deletes *but* and *now*, so "nothing but positive things" became "nothing positive thing", which reads as negative. |
| 10 | 12280 | 0.0007 | "Not the best accoustics, but still a fairly nice venue. This was the most bizarre concert I've ever attended. It's not the venue's fault, but it was a weird experience." | Sentiment about something else | The opinion about the venue is a few short sentences at the start. The rest of this 395-word review is a story about a rude couple in the audience. The model reacts to the angry story, not to what is being rated. |

## C. Near-threshold errors
_The model was unsure: P(pos) is almost exactly 0.5._

| # | Test idx | P(pos) | Snippet | Error type | What went wrong |
|---|---|---|---|---|---|
| 11 | 4169 | 0.5005 (true neg) | "Well it's over when they want it to be underwear night 11-to whenever they feel it's over" | Too little signal | This is the whole review. After cleaning only 7 tokens are left and none is an opinion word. 41% of the attention goes to *underwear*. A 0.5 output is a fair answer here. |
| 12 | 1772 | 0.5005 (true neg) | "Definitely not one of the nicer Hiltons I've stayed at, but close to the airport...so its convenient and the inside of the place is nice enough." | Mixed review | Good points (convenient, nice enough) and bad points (dark parking lot, bad area) are balanced. The *but* that separates them is deleted by my stopword list. |
| 13 | 29447 | 0.4993 (true pos) | "Great place to feed your caffeine addiction. … Not bad at all, but the sandwich was thrown together … There were at least 2 flies on the chocolate croissants." | Mixed review | Praise for the coffee, complaints about the food. The model's attention is split between *great* and *stale*. |
| 14 | 8723 | 0.4991 (true pos) | "scored two pairs of \$35 sweats for \$12 each, yea!!! … this place is hit and miss but i have always been able to find a decent deal" | Mixed review | The good news is told through prices and slang. "\$35 sweats for \$12" becomes "35 sweat 12". What is left sounds negative: *not great deal*, *miss*, *dumpy*. |
| 15 | 5190 | 0.5017 (true neg) | "it's time for a quick update, and a slight downgrade. … + Staff. Always friendly, always helpful. The biggest reasons for the downgrade? Size, produce, and pricing." | Mixed review | A long list of 5 cons marked "-" and 4 pros marked "+". Cleaning removes the "+" and "-" marks, so the model sees praise and complaints with nothing saying which side wins. |

## D. Slice-specific failures
_Worst slice: reviews where more than 10% of the tokens are unknown to the model (`<UNK>`). There are 231 of them and experimental_2 gets 14.72% wrong, against 4.67% overall. All 5 below are true negative, predicted positive._

| # | Test idx | P(pos) | Snippet | Error type | What went wrong |
|---|---|---|---|---|---|
| 16 | 1256 | 0.9990 | "The Calamari was delicious! … it didn't taste very fresh and had a rubbery texture … what a disappointment … the prices here are super steep" | Unknown words | The review has 9 photo links, which turn into about 100 junk tokens. It is 350 tokens after cleaning and my limit is 200, so the middle 150 are cut. Every complaint about the food (*not fresh, rubbery, mushy, disappointment*) is in the cut part. The praise at the start (*delicious*) survives. |
| 17 | 36104 | 0.9985 | "Das HubRaum ist für mich die Sommerkneipe in Durlach. … fandes jedesmal alles prima. Auch das Essen war immer lecker" | Unknown words | German. The model cannot read it. The text is mostly praise ("everything was great every time", "the food was always tasty") with one complaint at the end, so the negative label is also debatable. |
| 18 | 35963 | 0.9968 | "Interessante Gestaltung, mitten in der Stadt, unfreundliche Bedienung und ekelhafte Getränke, ein Stern ist da fast zu viel." | Unknown words | German: "unfriendly service and disgusting drinks, one star is almost too many". The model's top attention word is *fast*. In German it means "almost". In English it is the word for quick service. |
| 19 | 19328 | 0.9952 | "On n'y va QUE pour la section fruits et légumes, toujours bien remplie. Sinon, c'est sale!" | Unknown words | French: "otherwise, it's dirty!". The model attends to *bien, remplie, toujours*. The word *sale* means "dirty" in French but is a normal shopping word in English. |
| 20 | 27150 | 0.9923 | "Die Cocktails waren einen Besuch wert, alles andere war eher enttäuschend bei unserem Besuch." | Unknown words | German: "everything else was rather disappointing". The model's top attention words include *ideal* and *freundliche* ("friendly"). The word for "disappointing" is broken into two junk pieces, "entt" and "u00e4uschend", by my cleaning step (see pattern 5). |

## Patterns across the 20 errors

1. **Most errors are hard reviews, not random mistakes.** 7 of the 20 are mixed reviews, and 4 of the 5 near-threshold errors are mixed. A probability near 0.5 on those is the model being unsure for a good reason.
2. **Some "errors" are label problems.** In #1 and #6 the text plainly disagrees with the label. In #3, #4, #7 and #17 the label is debatable. No model can reach 100% on this test set.
3. **My stopword list deletes words that carry the meaning.** *but, than, other, now, before* are removed as stopwords. In #9, "nothing but positive things" became "nothing positive thing". In #2, "every other froyo joint than this one" became "every froyo joint one". Test reviews that contain "but" have a 5.16% error rate (n = 20,227). The rest have 4.11% (n = 17,773).
4. **The model reacts to strong words, not to what they are about.** In #8, #9 and #10 the angry words are aimed at other reviewers, old owners and other customers.
5. **Text the model cannot read still gets a confident answer.** The five slice errors all have P(pos) above 0.99. Part of this is caused by my own cleaning step. The dataset stores accented letters as codes like `ä`. My cleaner drops the backslash, so "enttäuschend" turns into "entt" and "u00e4uschend". 879 test reviews (2.3%) contain these codes and their error rate is 7.39%. Across the whole slice the errors go both ways (19 false positives, 15 false negatives), so the model is not simply biased towards positive.

## Proposed testable fix

**Fix:** stop deleting contrast and comparison words. Take *but, than, other, now, before, only, too* out of the stopword list, the same way negations (*not, never, no*) are already kept. Change nothing else.

**Hypothesis:** these words tell the model which half of a mixed review matters ("X but Y"), who is being compared ("better than"), and what is past or present ("now"). If the model can see them, the BiLSTM and the attention layer can learn to give more weight to the part after "but". This targets the mixed, indirect-negative and past-versus-now errors, which are 10 of the 20 above.

**How to test it:**
1. Retrain experimental_2 with the same config and seed. Only the stopword list changes.
2. Evaluate on the same 38,000 test reviews.
3. Compare the old and new predictions:
   - error rate on reviews that contain "but" (now 5.16%, n = 20,227): should go down
   - error rate on reviews without "but" (now 4.11%, n = 17,773): should not go up
   - overall error rate (now 4.67%)
   - paired McNemar test, old model against new model: the change counts as real if p < 0.05
4. Spot check: #2, #9 and #13 should move towards the correct label.

One caution: reviews with "but" are harder anyway because they are mixed. The 5.16% against 4.11% gap does not prove the fix will work. The test above is what would show it.

**A second, smaller fix for slice D:** decode the `ä`-style codes before cleaning, and replace each link with a single `<url>` token. To test it, compare the error rate on the 879 reviews that contain those codes (now 7.39%) and on the high-unknown slice (now 14.72%, n = 231). This does not make the model understand German or French. Those reviews would need a language check that flags them.

**What cannot be fixed in the model:** label noise (#1, #6). The text and the label disagree, so the model's answer is the sensible one.
