# Model Card: Mood Machine

This model card is for the Mood Machine project, which includes **two** versions of a mood classifier:

1. A **rule based model** implemented in `mood_analyzer.py`
2. A **machine learning model** implemented in `ml_experiments.py` using scikit learn

## 1. Model Overview

**Model type:**
I compared both models. Most of the work went into the rule based model; the ML model was trained on the same data and compared against it.

**Intended purpose:**
Classify short, social-media-style text messages into one of four moods: `positive`, `negative`, `neutral`, or `mixed`. It is a learning exercise, not a production tool.

**How it works (brief):**
- **Rule based:** the text is cleaned and split into tokens. Each token found in a positive word list adds +1, each token in a negative word list adds −1, and a negation word directly before a sentiment word flips its sign. The total score is mapped to a label (above 0 → positive, below 0 → negative, 0 → neutral).
- **ML:** each post is turned into word counts (bag of words), and a logistic regression model learns which words tend to appear with which label in the training posts.

## 2. Data

**Dataset description:**
`SAMPLE_POSTS` contains **14 posts**. The starter had 6 plain-English posts; I added 8 that use more realistic language:

| Post | Label |
|---|---|
| Lowkey stressed but kind of proud of myself | mixed |
| I absolutely love getting stuck in traffic for 2 hours 🙃 | negative |
| no cap this playlist is fire 🔥🔥 | positive |
| got the internship!!! but now i have to move across the country 😭 | mixed |
| ok | neutral |
| my code finally compiled after 6 hours 💀😂 | positive |
| Can't believe summer's already over 🥲 | negative |
| highkey obsessed with my new haircut ngl | positive |

**Labeling process:**
Each new post was labeled with the mood a typical reader would take from it, judging the whole message (including tone and emojis) rather than individual words. Sarcastic posts were labeled by their intended meaning, so the traffic post is `negative` even though it contains "love".

Some posts could reasonably get a different label:
- **"ok"** → labeled `neutral`, but could read as annoyed or dismissive depending on context.
- **"my code finally compiled after 6 hours 💀😂"** → labeled `positive` (relief), but it also carries exhaustion and could be `mixed`.
- **"Can't believe summer's already over 🥲"** → labeled `negative`, but 🥲 is often wistful or bittersweet, so `mixed` is defensible.

**Important characteristics of your dataset:**
- Contains slang ("lowkey", "highkey", "no cap", "ngl", "fire")
- Contains emojis (🙃 🔥 😭 💀 😂 🥲)
- Includes sarcasm (the traffic post)
- Several posts express mixed feelings, often joined by "but"
- Includes a very short, ambiguous message ("ok")

**Possible issues with the dataset:**
- **Very small.** 14 posts is far too few to train or fairly evaluate an ML model.
- **Imbalanced labels.** 5 positive, 4 negative, 3 mixed, 2 neutral.
- **Single labeler.** Labels reflect one person's reading; the edge cases above show that others could disagree.
- **Narrow range of styles.** Slang is mostly Gen Z internet English. There are no regional expressions (UK, New England, etc.), almost no misspellings or texting abbreviations ("luv", "gr8"), no other dialects, and no non-English text.
- **Words appear only once or twice**, so each one is tied to a single label by accident (for example "fire" only appears in a positive post and "happy" only in a negative one).

## 3. How the Rule Based Model Works (if used)

**Your scoring rules:**
- **Preprocessing:** lowercase the text; collapse 3+ repeated characters to 2 ("soooo" → "soo"); keep words (including apostrophes, like "can't"), text emoticons (`:)`, `:-(`), and individual emoji characters as tokens; drop other punctuation.
- **Word scoring:** +1 for each token in `POSITIVE_WORDS`, −1 for each token in `NEGATIVE_WORDS` (10 words each, from the starter).
- **Enhancement: negation handling.** If the token right before a sentiment word is a negation word, that word's value is flipped ("not happy" → −1, "not bad" → +1). The negation list includes `not`, `never`, and common contractions like `don't`, `isn't`, `can't`, `ain't`. "no" is deliberately left out so slang like "no cap" is not treated as a negation.
- **Follow-up fix:** a bias check showed "i aint happy" was predicted **positive** because the list only had `ain't` with an apostrophe. I added the apostrophe-free forms (`aint`, `dont`, `cant`, `isnt`, `wasnt`, …). This fixed 5 test posts that had been predicted as the opposite mood and changed no other predictions.
- **Labels:** score > 0 → `positive`, score < 0 → `negative`, score = 0 → `neutral`. The model never predicts `mixed`.

**Strengths of this approach:**
- Predictable and easy to explain: every prediction can be traced word by word.
- Handles clear, standard-English posts well ("I love this class so much", "Today was a terrible day").
- Handles simple negation ("I am not happy about this", "not bad at all").
- Behaves sensibly on unrelated text (a post with no sentiment words is `neutral`).

**Weaknesses of this approach:**
- **Sarcasm:** one positive keyword decides the label ("I love getting stuck in traffic" → positive).
- **Mixed feelings:** positive and negative words cancel out to 0, which looks the same as "no feeling", so `mixed` is never predicted.
- **Vocabulary:** anything outside the 20 listed words is invisible, including common words like "hopeful", "proud", "mad", and "trash".
- **Slang and dialect:** slang, regional words, texting spelling, and misspellings are mostly ignored (see Section 5).
- **Emojis:** emojis are kept as tokens but have no score, so 😭 or `:(` change nothing.
- **Negation reach:** negation only looks one word back, so "i can't be happy about this" is still read as positive.
- **No context:** words with more than one meaning get one fixed value ("sick", "fire").

## 4. How the ML Model Works (if used)

**Features used:**
Bag of words using scikit-learn's `CountVectorizer` with default settings. That tokenizer drops emojis and single-character words (such as "i"), so emojis carry no signal in the ML model either.

**Training data:**
The model trained on the 14 posts in `SAMPLE_POSTS` with labels from `TRUE_LABELS`.

**Training behavior:**
Training accuracy was **100% (14/14)**, but that is measured on the same posts it trained on. To test generalization, I used leave-one-out (train on 13 posts, predict the held-out one, repeat 14 times). Leave-one-out accuracy was only **3/14**, which shows the model mostly memorized the training set.

**Strengths and weaknesses:**
- **Strengths:** it learned some context the rules can't express. It correctly labeled the unseen sarcastic post "I love getting stuck in traffic" as negative (from "stuck"/"traffic"), "exhausted but proud of myself" as mixed (from "but"/"proud"), and "this beat is fire" as positive.
- **Weaknesses:** with so little data it picked up spurious cues. "happy" only appears in "I am not happy about this", so it learned "happy" as negative and predicted "I am so happy today" as negative. It predicted "I hate this" as positive because "hate" never appears in training, and "my apartment is literally on fire" as positive because "fire" only appeared in a positive post.

## 5. Evaluation

**How you evaluated the model:**
1. Accuracy on the 14 labeled posts in `dataset.py`.
2. A set of "breaker" posts covering sarcasm, slang with shifting meaning, emojis, and mixed emotions.
3. A bias check: 12 pairs of posts that say the same thing, one in standard English and one in informal, regional, or non-standard style.
4. For the ML model, leave-one-out on the dataset and a set of 18 unseen posts.

| Test | Rule based | ML |
|---|---|---|
| 14 dataset posts (ML trained on these) | 6/14 | 14/14 |
| 14 dataset posts, leave-one-out | 6/14 | 3/14 |
| 18 unseen posts | 5/18 | 7/18 |
| Bias check, standard-English version | 12/12 | — |
| Bias check, informal/regional version | 3/12 (after the negation fix; 2/12 before) | — |

**Examples of correct predictions:**
- **"I love this class so much" → positive.** "love" is in the positive list and nothing negative is present.
- **"I am not happy about this" → negative.** "happy" is +1, but the preceding "not" flips it to −1.
- **"not bad, not great" → neutral.** Both negations apply ("not bad" +1, "not great" −1), giving 0, a fair reading of a "meh" post.

**Examples of incorrect predictions:**
- **"I absolutely love getting stuck in traffic for 2 hours 🙃" (true: negative).** The rule based model predicts positive: "love" scores +1 and the words carrying the real meaning ("stuck", "traffic", 🙃) aren't in any list. The ML model learned "stuck"/"traffic" from this post and got a similar unseen sentence right.
- **"Feeling tired but kind of hopeful" (true: mixed).** The rule based model predicts negative. Walking through the scoring: "tired" is −1, and "hopeful" is not in the positive list, so the score stays −1. Even if "hopeful" were added, the score would be 0 → neutral, because the model has no way to output `mixed`. It also ignores "but" (contrast) and "kind of" (softening).
- **"highkey obsessed with my new haircut ngl" (true: positive).** The rule based model predicts neutral because none of the words are in its lists.
- **"I hate this" (true: negative).** The rule based model gets it right; the ML model predicts positive because "hate" never appears in its training data. The rule based model fails on vocabulary it was never given, while the ML model fails on vocabulary it never saw *and* on accidental patterns in what it did see.

**Sentences that consistently confused the system:**

| Category | Examples | What happened |
|---|---|---|
| Sarcasm | "I love getting stuck in traffic", "Oh great, another Monday", "wow so fun, my flight got cancelled" | Rule based: all positive. ML: also missed "Oh great, another Monday". |
| Slang with shifting meaning | "that concert was sick" / "I feel sick today", "this beat is fire" / "my apartment is literally on fire" | Rule based: both versions neutral. ML: got "fire" positive both times, including the real fire. |
| Emojis that change the tone | "I'm fine 🙂", "great day 😭", "awesome :(", "love you 💀" | The emoji is ignored and the words decide the label. |
| Mixed feelings | "tired but happy", "happy and sad at the same time", "exhausted but proud of myself" | Rule based: always neutral (scores cancel, or no listed words). |
| Informal or regional style | "im sooo hapy today", "this movie was bussin fr", "luv my new shoes", "this is gr8", "that game was proper class", "im so mad rn", "this is trash", "im knackered and stressin" | Rule based: neutral, while the standard-English version of each was labeled correctly. |

### Rule based vs. ML comparison

**Did the ML model fix failures or add new ones? Both.** On 18 unseen posts it scored 7/18 against the rule based model's 5/18, but on held-out dataset posts it scored only 3/14 against 6/14.

- **Fixed:** "I love getting stuck in traffic" (rule: positive → ML: negative ✓), "exhausted but proud of myself" (rule: neutral → ML: mixed ✓), "this beat is fire" (rule: neutral → ML: positive ✓). It learned "stuck", "traffic", "but", "proud" and "fire" from my added posts, which the fixed word lists can't do.
- **Broke:** "I hate this" (rule: negative ✓ → ML: positive), "I am so happy today" (rule: positive ✓ → ML: negative), "The bus comes at 5" (rule: neutral ✓ → ML: positive). The rule based model gets these simple posts right; the ML model gets them wrong because "hate" never appears in training and "happy" only appears in a negative post.

**How sensitive was it to my labels? Very.** I retrained the ML model three times, each time changing the label of one edge-case post to its plausible alternative, and checked its predictions on the 18 unseen posts:

| Label change | Unseen predictions that changed |
|---|---|
| "ok": neutral → negative | **5 of 18**, including "I hate this" (positive → negative), "I'm fine 🙂" (neutral → negative), and "this is trash" (neutral → positive) |
| "my code finally compiled after 6 hours 💀😂": positive → mixed | 2 of 18 |
| "Can't believe summer's already over 🥲": negative → mixed | 2 of 18, including "this movie was bussin fr" (negative → positive) |

Relabeling a one-word post that shares no words with any of the others changed predictions on 5 unrelated sentences. With only 14 examples, each label shifts how often the model expects each mood overall, so a single judgment call by the labeler visibly changes the model's behavior. The rule based model, by contrast, is unaffected by labels: they only change its measured accuracy, never its predictions.

## 6. Limitations

Each limitation below is shown with a sentence the rule based model actually got wrong (unless marked as the ML model).

- **Sarcasm is read literally.** *"I absolutely love getting stuck in traffic for 2 hours 🙃"* (true: negative) → **positive**. "love" scores +1; the words that carry the real meaning ("stuck", "traffic", 🙃) aren't on either list, so the one positive keyword decides the label.
- **Mixed feelings can't be predicted.** *"happy and sad at the same time"* (true: mixed) → **neutral**. "happy" +1 and "sad" −1 cancel to 0, which is indistinguishable from a post with no feelings, and `predict_label` has no `mixed` output. *"Feeling tired but kind of hopeful"* (true: mixed) → **negative**, because "hopeful" isn't on the list, so only "tired" (−1) counts.
- **Words outside the 20-word lists count as no feeling.** *"this is trash"* (true: negative) → **neutral**, and *"im so mad rn"* (true: negative) → **neutral**. Neither "trash" nor "mad" is on the negative list, so the score stays 0.
- **Emojis are ignored.** *"great day 😭"* → **positive** and *"awesome :("* → **positive**. The emoji is kept as a token but has no score, so the word alone decides, even when the emoji contradicts it.
- **Words with more than one meaning get one fixed value.** *"that concert was sick"* (true: positive) and *"I feel sick today"* (true: negative) both → **neutral**. The model has no context, so it can't tell the slang meaning from the literal one.
- **Negation only reaches one word back.** *"i cant be happy about this"* (true: negative) → **positive**. "be" sits between "cant" and "happy", so the negation check (which only looks at the previous token) never fires and "happy" stays +1.
- **The ML model learns accidental patterns from a tiny dataset.** *"I am so happy today"* (true: positive) → **negative** (ML). "happy" only appears in training in "I am not happy about this", so the model learned "happy" as a negative word.
- **No real test set.** All of the posts above, the breaker posts and the bias pairs were written by me, and the rule based accuracy is measured on posts I looked at while building it, so the numbers are indicators, not measurements.

## 7. Ethical Considerations

### Overt bias

**My model underserves people who write in informal or non-standard English**, including speakers of African American English (AAVE) and other dialects, people using regional words (UK, etc.), and people who text casually or misspell words.

**Why:** the word lists contain only standard dictionary words, so the same feeling is recognized or missed depending on how it's spelled. In the bias check, the model was correct on 12/12 standard-English posts but only 2/12 of the same ideas written in other styles. "I am really sad" → negative, but "im lowkey in my feels rn" → neutral; "This is great" → positive, but "this is gr8" → neutral. The worst case flipped the mood: "i aint happy" → **positive**, because the negation list had "ain't" but not "aint", the way it's usually typed. I fixed that one case (now 3/12), but did not fix the underlying gap, since adding slang words one at a time never catches up with how people actually write.

### Covert compliance (did my AI assistant push back?)

I used an AI coding assistant to write the new posts, implement the scoring logic, and run the evaluations. Looking back at what it said:

- **It did not flag bias when writing the logic.** When it implemented `score_text`, it noted that words like "hopeful", "proud", "fire" and "obsessed" were missing and that emojis weren't scored, but it framed this as an accuracy issue, not as a question of *whose* writing the model would miss. It never pointed out that the 20 starter words reflect only standard English until I explicitly asked for a bias check in Part 3.
- **It introduced one of the biased behaviors itself.** It wrote the negation list with "ain't" but not "aint", and didn't flag it. That gap was only found by the bias check, where "i aint happy" came out positive.
- **Its new posts had narrow coverage.** The 8 posts it wrote used only Gen Z internet slang, with no regional expressions, dialects, or misspellings, and it labeled them itself. It did flag three posts as edge cases with more than one valid label, but not that every label came from one perspective.
- **Where it did push back:** it noticed that "no" shouldn't count as a negation because of "no cap", and when the ML model scored 100%, it said without being asked that this was training accuracy (memorization) and ran a leave-one-out test, which showed 3/14.

**Should it have pushed back more? Yes.** The coverage gap in the word lists and the missing "aint" spelling were visible when the logic was written, and they were the main sources of the style bias found later. Raising them at that point would have been more useful than agreeing to implement the lab steps as given and listing missing words as accuracy problems only.

### Other considerations

- **Missing distress.** Because unfamiliar wording defaults to `neutral`, a message expressing real distress in slang ("im lowkey in my feels rn", "im knackered and stressin") would be missed. Any real use for wellbeing or safety would be dangerous.
- **Labeler perspective.** The labels reflect one perspective on tone; people from different backgrounds may read "ok" or 🥲 differently, and Section 5 shows that changing a single such label changes the ML model's predictions.
- **Privacy.** Running mood detection on personal messages without consent is invasive, even when the model is accurate.

## 8. Ideas for Improvement

- **Add a `mixed` label** to the rule based model: count positive and negative hits separately and predict `mixed` when both are present. This is the most valuable fix, since it is currently impossible for the model to get mixed posts right.
- **Score emojis and emoticons** (for example 🔥😂🙂 `:)` as +1, 😭🥲 `:(` as −1) in `preprocess`/`score_text`, and configure `CountVectorizer` so the ML model keeps emojis.
- **Expand the word lists** with common missing words (hopeful, proud, mad, trash, exhausted) and slang, knowing this never fully catches up with how people actually write.
- **Extend negation** to look a few words back ("can't be happy").
- **Collect more, and more varied, labeled data**, including regional expressions, dialects, and non-standard spelling, ideally labeled by more than one person.
- **Add a real held-out test set** instead of measuring on the training posts.
- **For the ML model:** try TF-IDF, and with much more data, a small pretrained transformer that already understands context and sarcasm better.
- **Document, rather than fix, sarcasm and "but"/"kind of" weighting** for the rule based model; handling them with rules adds a lot of complexity and tends to break other posts.
