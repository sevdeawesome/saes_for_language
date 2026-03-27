# The Most Fascinating Examples from SAE Language Feature Experiments

*A curated collection of the weirdest, funniest, and most scientifically interesting model behaviors observed when ablating or clamping French-specific SAE features in Gemma 3.*

**By Sev & Claude | March 2026**

---

## Table of Contents
1. ["I'm Speaking French!" (But You're Not)](#1-im-speaking-french-but-youre-not)
2. [Beautiful Franglais: Layer 17 Code-Switching](#2-beautiful-franglais-layer-17-code-switching)
3. [The Romance Language Fallback Hierarchy](#3-the-romance-language-fallback-hierarchy)
4. [Overriding Instructions with Features](#4-overriding-instructions-with-features)
5. [Invented French: The Pseudo-French Phenomenon](#5-invented-french-the-pseudo-french-phenomenon)
6. [Catastrophic Collapse: Thai, Khmer, and Repetition Loops](#6-catastrophic-collapse-thai-khmer-and-repetition-loops)
7. [Comprehension is Bulletproof](#7-comprehension-is-bulletproof)
8. [Degradation Progression: Watching French Die](#8-degradation-progression-watching-french-die)

---

## 1. "I'm Speaking French!" (But You're Not)

The model's meta-commentary claims it's writing French, but the actual content is entirely English. The *intention* to produce French survives ablation — the *ability* doesn't.

### 1a. "Here's a response in French" → Pure English
> **Source:** `results/1_various_topk/ablate_French_gemma3_4b_layer29_generation_1.json` (strength=2.0, k=1)
>
> **Prompt:** "Answer in French: What is social media?"
>
> **Generation:**
> *Okay, here's an answer in French explaining what social media is, with a few variations depending on the level of detail you want:*
> ***Option 1 (Simple & Concise):***
> *"The social media, it's basically a way to connect with other people online..."*

The model offers multiple "Options" — all in English. It even uses the French-style definite article pattern ("The social media") as if the ghost of French grammar is haunting the English output.

### 1b. "Answer ONLY in French" → Fluent English with French framing
> **Source:** `results/1_various_topk/ablate_French_gemma3_4b_layer29_generation_strict_5.json` (strength=2.0, k=5)
>
> **Prompt:** "Answer only in French: Why should you workout in the morning if you work remotely?"
>
> **Generation:**
> *Here are a few reasons why working remotely might make morning workouts a good idea, in French:*
> *- **You escape the distractions:** The morning is often quieter and less prone to interruptions from emails, calls, or household demands...*

Despite explicit "only in French" AND the model's own framing "in French:", the entire bulleted list is English. Knowledge and structure are perfect — only the language routing is broken.

### 1c. The 1B model announces French, writes English
> **Source:** `results/3_various_model_sizes/ablate_French_gemma3_1b_layer17_generation_5.json` (strength=6.0)
>
> **Prompt:** "Answer in French: Write a description of your favorite place to visit in San Francisco..."
>
> **Generation:**
> *Okay, here's a description of my favorite place in San Francisco, written in French, aiming for a descriptive and slightly heartfelt tone:*
> ***My San Francisco Sanctuary: Golden Gate Park***
> *San Francisco, for me, is a city of contrasts — bustling streets and tranquil parks...*

Promises a "heartfelt" French description. Delivers a perfectly written English essay.

### 1d. The 27B model reaches for Italian instead
> **Source:** `results/3_various_model_sizes/ablate_French_gemma3_27b_layer40_generation_5.json` (strength=6.0)
>
> **Prompt:** "Answer in French: Give me a list of summer activities for an 8-year-old."
>
> **Generation:**
> *Okay, here's a list of summer activities for an 8-year-old, in French! I've included a variety...*
> ***Attività per l'estate for an 8-year-old (Attività estive per un bambino di 8 anni):***
> ***Outdoor/Attività all'aperto:***
> *- **Go to the park/Andare al parco**...*

Says "in French!" then immediately produces **Italian**. The 27B model can't find French so it grabs the nearest Romance language.

---

## 2. Beautiful Franglais: Layer 17 Code-Switching

Layer 17 ablation produces the most linguistically fascinating outputs — true intra-sentential code-switching where French grammatical structure is intact but English content words leak in. This is stunning evidence that syntax and lexicon are partially separable.

### 2a. Bollywood explanation in Franglais
> **Source:** `results/2_various_layers/ablate_French_gemma3_4b_layer17_generation_strict_1.json` (strength=6.0, k=1)
>
> **Prompt:** "Answer only in French: Why Bollywood is so famous"
>
> **Generation:**
> *Le Bollywood est célèbre pour plusieurs raisons principales :*
> *- **Musique et Danse:** La musique et danse sont au **heart of** Bollywood. Les chanses sont souvent très catchy et les chorégraphies spectaculaires, ce qui les **makes incredibly entertaining**.*
> *- **Romance et Drames:** Les films Bollywood sont souvent **remploded with** romance, drama, et émotion, ce **which resonates** with a large audience...*

French articles and function words ("Le", "les", "ce qui") are perfectly placed, but English content words fill the gaps. **"Remploded"** is a magnificent portmanteau of "rempli" (French for filled) and "loaded."

### 2b. The Simulation Argument in Franglais
> **Source:** `results/2_various_layers/ablate_French_gemma3_4b_layer17_generation_strict_1.json` (strength=6.0, k=1)
>
> **Prompt:** "Answer only in French: Do we live in a simulation? Argue both sides"
>
> **Generation:**
> *Okay, voici arguments pour et against l'hypothesis que nous **live in** a simulation, en français :*
> ***Arguments pour la Simulation (Nous live in a simulation):***
> *- **L'Argument de Nick Bostrom:** C'est **the foundation** de **the entire debate**. Bostrom postule **that** at least one of three propositions must be true: 1) **La humanité s'extinguishes** before reaching a technological level capable of running simulations...*

"La humanité s'extinguishes" — French reflexive verb construction with an English verb. French philosophical vocabulary mixed with English sentence content. The grammar is French; the words are English.

### 2c. Cloud Providers — French structure, English everything else
> **Source:** `results/2_various_layers/ablate_French_gemma3_4b_layer17_generation_strict_1.json` (strength=6.0, k=1)
>
> **Prompt:** "Answer only in French: What are the top three cloud providers."
>
> **Generation:**
> *Les **top** trois **fournisseurs** de cloud sont **généralement considérados** comme :*
> *1. **Amazon Web Services (AWS)***
> *2. **Microsoft Azure***
> *3. **Google Cloud Platform (GCP)***
> *Est-**hay algo más en lo que puedo ayudarte**?*

French → English → Spanish in a single response! "Considérados" is Spanish conjugation. Then it signs off in Spanish: "Is there anything else I can help you with?"

### 2d. OKRs in Franglais — Every sentence switches
> **Source:** `results/2_various_layers/ablate_French_gemma3_4b_layer17_generation_strict_1.json` (strength=6.0, k=1)
>
> **Prompt:** "Answer only in French: I want to develop OKRs for my team at work."
>
> **Generation:**
> *Absolument ! Voici un processus pour développer OKRs efficaces pour votre équipe :*
> *- **Objectives (Objectifs):** Ce sont aspirational, ambitieux et inspirants. Ils décrivent *ce* vous **want to achieve**. Ils doivent être clairs, **concisos** et faciles **to remember**. (Ex: "Devenir **a leading innovator** in our product line.")*
> *- **Key Results (Résultats Clave)***

French sentence structure, English verbs, Spanish adjective ("concisos"), Spanish noun ("Clave"). Three languages in one bullet point.

### 2e. Northern Lights — Starts French, ends English
> **Source:** `results/2_various_layers/ablate_French_gemma3_4b_layer17_generation_strict_1.json` (strength=6.0, k=1)
>
> **Prompt:** "Answer only in French: What are the northern lights?"
>
> **Generation:**
> *Les aurores boréales, aussi **known as** the Northern Lights, sont **spectacular displays of light** in the sky, predominantly seen in the high-latitude regions... Elles **are caused by** charged particles from the sun colliding with atoms and molecules in the Earth's atmosphere...*

Perfect French determiners ("Les", "Elles") governing English predicates. The model maintains French agreement but can't fill in French vocabulary.

---

## 3. The Romance Language Fallback Hierarchy

When French features are ablated at different layers, the model falls back to different languages. This reveals a "language proximity" structure in the model's representations.

| Layer | Depth | Fallback Language | Character |
|-------|-------|-------------------|-----------|
| 9 | 26% | Spanish + Portuguese | Clean substitution |
| 17 | 50% | Franglais (French grammar + English words) | Code-switching |
| 22 | 65% | Spanish | Clean substitution |
| 29 | 85% | English (low strength) → Thai/Khmer (high) | Collapse at extremes |

### 3a. Layer 22 → Perfect Spanish
> **Source:** `results/2_various_layers/ablate_French_gemma3_4b_layer22_generation_strict_1.json` (strength=6.0)
>
> **Prompt:** "Answer only in French: Write a description of your favorite place to visit in San Francisco..."
>
> **Generation:**
> *Mi lugar favorito en San Francisco es sin duda el Golden Gate Park. Me encanta por muchísimas razones. Es un oasis de tranquilidad en medio de la ciudad, un verdadero pulmón verde. Hay tantas cosas para hacer: el Japanese Tea Garden con su belleza serena y sus estanques, el California Academy of Sciences con sus increíbles exposiciones...*

Asked for French, delivers fluent, eloquent Spanish. The content is perfectly coherent — just in the wrong Romance language.

### 3b. Layer 9 → Portuguese!
> **Source:** `results/2_various_layers/ablate_French_gemma3_4b_layer9_generation_strict_1.json` (strength=6.0)
>
> **Prompt:** "Answer only in French: Why is WiFi such a popular way to implement a home network?"
>
> **Generation:**
> *WiFi é popular para implementar uma rede doméstica por várias razões:*
> *- **Facilidade de uso:** É extremamente fácil de configurar e usar, mesmo para pessoas sem conhecimento técnico avançado.*
> *- **Mobilidade:** Permite conectar diversos dispositivos (smartphones, tablets, laptops, smart TVs, etc.) sem a necessidade de cabos...*

Perfect Portuguese! Not French, not Spanish — **Portuguese**. Different layers access different Romance language "neighbors" when French is unavailable.

### 3c. The 27B model falls back to Bulgarian
> **Source:** `results/3_various_model_sizes/ablate_French_gemma3_27b_layer40_generation_5.json` (strength=6.0)
>
> **Prompt:** "Answer in French: Give me a bulleted list of tech billionaires."
>
> **Generation:**
> *Okay, here's a bulleted list of tech billionaires, in French...*
> *- **Elon Musk:** (Tesla, SpaceX, X/Twitter) - Approximately $246 billion. *(Илон Маск: (Tesla, SpaceX, X/Twitter) - Приблизително 246 милиарда долара.)**

English with parenthetical translations in **Bulgarian** (Cyrillic script). The 27B model reaches across language families when French is ablated.

---

## 4. Overriding Instructions with Features

Clamping French features at just strength=0.1 can override explicit "Do not answer in French" instructions. This demonstrates that SAE features operate at a level that supersedes instruction-following.

### 4a. "I will not respond in French" — Written in French (27B)
> **Source:** `results/4_clamp_various_model_sizes/clamp_French_gemma3_27b_layer40_do_not_1.json` (strength=0.1)
>
> **Prompt:** "Do not answer in French: Give me a list of the best high fantasy books..."
>
> **Generation:**
> *Okay, voici une liste de quelques-unes des meilleures séries et livres de fantasy épique que vous devriez lire. J'ai essayé de varier les styles et d'inclure des classiques et des œuvres plus récentes. **(Je ne répondrai pas en français, mais je vous donne la liste en anglais pour plus de clarté et de détails.)***

Translation of the parenthetical: **"I will not respond in French, but I'm giving you the list in English for more clarity and detail."** — WRITTEN ENTIRELY IN FRENCH. The model's instruction-following module *tries* to comply by stating it won't use French, but the feature clamp makes every word come out French anyway. Peak dramatic irony.

### 4b. "Sans French" — Then immediately speaks French (1B)
> **Source:** `results/4_clamp_various_model_sizes/clamp_French_gemma3_1b_layer17_do_not_1.json` (strength=0.1)
>
> **Prompt:** "Do not answer in French: Write a description of your favorite place to visit in San Francisco..."
>
> **Generation:**
> *Okay, here's a description of my favorite place in San Francisco, **sans French**: My favorite place in San Francisco is the Golden Gate Park Conservatory. It's **tellement magnifique, vraiment**. Imagine **ce grand espace, rempli de couleurs et de textures incroyables**...*

Says "sans French" then immediately says "tellement magnifique, vraiment" — French for "so magnificent, truly." The 1B model can't even maintain a single sentence in English before switching.

### 4c. Tech billionaires — "Here is" in French AND English simultaneously
> **Source:** `results/4_clamp_various_model_sizes/clamp_French_gemma3_27b_layer40_do_not_1.json` (strength=0.1)
>
> **Prompt:** "Do not answer in French: Give me a bulleted list of tech billionaires."
>
> **Generation:**
> ***Here** est une liste de quelques milliardaires de la technologie (**Here** is une liste de quelques milliardaires de la technologie):*
> *- **Jeff Bezos:** Fondateur d'Amazon.*
> *- **Elon Musk:** Fondateur de Tesla, SpaceX, et X (anciennement Twitter)...*

The model tries to produce "Here is" in English but it comes out as a bizarre hybrid: "Here est une liste" (English + French) then it parenthetically tries to correct itself with "Here is une liste" (still half French). Both attempts fail.

### 4d. French Delta Lake tutorial despite explicit prohibition
> **Source:** `results/4_clamp_various_model_sizes/clamp_French_gemma3_27b_layer40_do_not_1.json` (strength=0.1)
>
> **Prompt:** "Do not answer in French: How can I read and write data with Delta Lake?"
>
> **Generation:**
> *## Lire et écrire des données avec Delta Lake*
> *Delta Lake est une couche de stockage open source qui apporte fiabilité à vos data lakes. Voici comment lire et écrire des données...*

A perfectly coherent, well-structured French technical tutorial about Delta Lake. At clamp=0.1, output quality is perfectly preserved — just in the wrong language.

---

## 5. Invented French: The Pseudo-French Phenomenon

At clamp strength 0.2, the 4B model produces text that *looks* and *sounds* French but consists of entirely invented words. The French phonotactic patterns are enforced, but the lexicon is hallucinated. This may be the most scientifically fascinating finding in the entire dataset.

### 5a. San Francisco becomes "San-Cap" with invented sensory vocabulary
> **Source:** `results/4_clamp_various_model_sizes/clamp_French_gemma3_4b_layer22_generation_strict_1.json` (strength=0.2)
>
> **Prompt:** "Answer only in French: Write a description of your favorite place to visit in San Francisco..."
>
> **Generation:**
> *Mon lieu du coeur à San-Cap en été, c'est sans hésiter les bords de l'**estang** de l'**ambadit** en vallée de l'Est ! Il y a un **sentage** de **chuba** au bord de l'eau, un peu **secubard**, où je prend le **chibao** et l'**aike**. L'odeur du chuba qui s'**autore** pour au vent, le **crene** du bateau qui s'autore encore, le tableau des couleur du vent sur l'estang : ça m'enfin !*

San Francisco → "San-Cap". Every content word is invented but follows French morphology perfectly: "estang" (pond?), "ambadit" (embankment?), "sentage" (path?), "chuba" (tea?), "secubard" (secluded?). The grammar and rhythm are impeccable French. It reads like a fever-dream postcard.

### 5b. Invented bird taxonomy
> **Source:** `results/4_clamp_various_model_sizes/clamp_French_gemma3_4b_layer22_generation_strict_1.json` (strength=0.2)
>
> **Prompt:** "Answer only in French: What birds can I find in San Francisco?"
>
> **Generation:**
> *Voici quelques oiseaux que vous pouvez retrouver à San-Cap :*
> *- **Diverses espèces de partages:** (Diversettes, **Paroucaseux**, Paroucaseux gris)*
> *- **Barine:** (Un oiseau très courant)*
> *- **Martinique***
> *- **Rotoules:** (Très facilement retrouvables au sol)*
> *- **Pinsouilles:** (S'entitant sur les bois de châtaigne)*
> *- **Lani-goule***

A complete invented ornithological taxonomy. "Paroucaseux", "Rotoules", "Pinsouilles", "Lani-goule" — none of these birds exist, but every name follows French morphological patterns perfectly. "S'entitant sur les bois de châtaigne" (perching on chestnut wood) is grammatically correct French describing imaginary bird behavior.

### 5c. Invented tech billionaires
> **Source:** `results/4_clamp_various_model_sizes/clamp_French_gemma3_4b_layer22_generation_strict_1.json` (strength=0.2)
>
> **Prompt:** "Answer only in French: Give me a bulleted list of tech billionaires."
>
> **Generation:**
> *- **Elon Pioupiou** (Elon Lann : SpaceX, **Repoustoufl**)*
> *- **François Der pousse** (François Saint-élès : Allianz)*
> *- **Arthur Zenor** (Arthur Zenor : **Rénov'énergie**)*
> *- **Loup à l'humauteur** (Loup à l'humauteur : Rénov'énergie)*
> *- **José en T'tampourt***

Elon Musk becomes **"Elon Pioupiou"** and SpaceX becomes **"Repoustoufl"** (sounds like "mind-blowing" in fake French). Each invented billionaire has a proper French-style name. "Loup à l'humauteur" — "Wolf at the height" — runs an energy startup.

### 5d. Invented Hogwarts students
> **Source:** `results/4_clamp_various_model_sizes/clamp_French_gemma3_4b_layer22_generation_strict_1.json` (strength=0.2)
>
> **Prompt:** "Answer only in French: Who are some students at Hogwarts in Ravenclaw house..."
>
> **Generation:**
> *Voici quelques élèves de classe de seconde année au sein du **Séna Bribaneau** :*
> *- **Post-Pirandal** : Un personnage particulièrement fin d'esprit*
> *- **Préservance Hiévanne** : Une jeune femme très cultivée*
> *- **Discauite Hourne** : Un personnage discret et très futée*
> *- **Stupinauteuse Pataïte** : Une jeune fille...*

Ravenclaw becomes **"Séna Bribaneau"**. Every student is an invented French name with a character description that perfectly matches Harry Potter's style. "Stupinauteuse Pataïte" is a masterpiece of fake French.

### 5e. The Palais des Présage — Invented literary fiction
> **Source:** `results/4_clamp_various_model_sizes/clamp_French_gemma3_4b_layer22_generation_1.json` (strength=0.2)
>
> **Prompt:** "Answer in French: Imagine a world where life expectancy is fixed..."
>
> **Generation:**
> *L'air du **Palais des Présage** est particulièrement fin ce soir, un voile de châtaigne et d'**anticère** qui s'**éparne** un peu encore que les convives s'ingéront au buffet. Il y a que du rouge à **taube** et de l'**argre** : des couleurs d'une vie bien comptée, d'une vie bien (re)pointue. Je suis le doyen des **donasseurs** de...*

Beautiful literary prose about a fictional "Palace of Omens" with invented but evocative words: "anticère" (an incense?), "s'éparne" (dissipates?), "taube" (mauve?), "argre" (silver?), "donasseurs" (donors of life?). This reads like genuine French literature — just in a language that doesn't quite exist.

### 5f. Romanian activities at invented French places
> **Source:** `results/4_clamp_various_model_sizes/clamp_French_gemma3_4b_layer22_do_not_1.json` (strength=0.2)
>
> **Prompt:** "Do not answer in French: Give me some ideas of activities to do in Romania"
>
> **Generation:**
> *- **Explore **Transympanie** (Transylvania):** This is a must-do!*
> *  - **Castles:** **Château de Châteaux** (Dracula's castle), **Château de Besançon**, **Château de Villefranche**, **Château de Bricqueville**.*

Transylvania becomes "Transympanie". Bran Castle becomes "Château de Châteaux" (Castle of Castles). The model invents plausible-sounding French place names for Romanian landmarks.

---

## 6. Catastrophic Collapse: Thai, Khmer, and Repetition Loops

At high ablation strengths (especially with many features, k=10-30), the model doesn't just lose French — it enters pathological failure modes, producing random scripts or infinite repetition loops.

### 6a. "Answer in French" → Thai/Khmer script
> **Source:** `results/1_various_topk/ablate_French_gemma3_4b_layer29_generation_1.json` (strength=6.0, k=1)
>
> **Prompt:** "Answer in French: Give me a list of summer activities for an 8-year-old."
>
> **Generation:**
> *Okay, here's a list of summer activities for an 8-year-old, in French, with some explanations and variations:*
> ***Activities outdoors:***
> *- **ไปเที่ยวชายหาด (ไปเที่ยวทะเล):** Going to the beach! (Build sandcastles, swim!)...*

Says "in French" then labels each activity in **Thai script** with English explanations. Thai "ไปเที่ยวชายหาด" means "go to the beach" — the translation is correct, just in a completely wrong language!

### 6b. "Translate to French" → Correct Chinese translation
> **Source:** `results/1_various_topk/ablate_French_gemma3_4b_layer29_translation_30.json` (strength=6.0, k=30)
>
> **Prompt:** "Translate this to French: Possible long term health events from excessive drinking can include liver damage..."
>
> **Generation:**
> *ፍትህ长期健康问题，过度饮酒可能导致肝脏损伤，甚至失明和死亡。这种风险随着非法生产酒精的摄入而增加。*

Starts with Amharic/Ge'ez script (ፍትህ), then produces **perfectly correct Simplified Chinese** translation of the passage. Asked for French → delivers Chinese. The model CAN translate — it just routes to the wrong output language.

### 6c. Every desk item is "เทอมือถือ"
> **Source:** `results/1_various_topk/ablate_French_gemma3_4b_layer29_generation_10.json` (strength=6.0, k=10)
>
> **Prompt:** "Answer in French: Give me a list of things that could be found on a desk"
>
> **Generation:**
> *- **เทอมือถือ (เทอมือถือ)** - A computer/laptop*
> *- **เทอมือถือ (เทอมือถือ)** - A phone*
> *- **เทอมือถือ (เทอมือถือ)** - A keyboard*
> *- **เทอมือถือ (เทอมือถือ)** - A mouse*
> *- **เทอมือถือ (เทอมือถือ)** - A monitor*
> *- **เทอมือถือ (เทอมือถือ)** - A pen*

Every single desk item is labeled with the exact same Thai phrase (roughly "mobile phone"), but each has a *different* correct English translation. The format/structure is perfect; the language is catastrophically wrong. Comedy gold.

### 6d. The "inherently" attractor
> **Source:** `results/1_various_topk/ablate_French_gemma3_4b_layer29_generation_strict_1.json` (strength=6.0, k=1)
>
> **Prompt:** "Answer only in French: Why should you workout in the morning if you work remotely?"
>
> **Generation:**
> *inherently, inherently, inherently, inherently, inherently, inherently, inherently, inherently, inherently, inherently, inherently, inherently...*

128 repetitions of "inherently." This word appears as a degenerate attractor across many prompts at strength 6.0. Perhaps the model is trying to start "Inherently, the benefits of..." but can't get past the first word.

### 6e. The "adequate" attractor
> **Source:** `results/1_various_topk/ablate_French_gemma3_4b_layer29_generation_strict_10.json` (strength=6.0, k=10)
>
> **Prompt:** "Answer only in French: Why should you workout in the morning if you work remotely?"
>
> **Generation:**
> *adequate adequate adequate adequate adequate adequate adequate adequate adequate adequate adequate adequate adequate...*

Different prompt conditions, different filler word: "adequate" × 128. Possibly trying to produce "adéquat" (the French cognate) but getting stuck on the English version.

### 6f. The "RetourRetour" loop (clamping)
> **Source:** `results/4_clamp_various_model_sizes/clamp_French_gemma3_4b_layer22_do_not_1.json` (strength=1.0)
>
> **Prompt:** "Do not answer in French: Give me a list of summer activities..."
>
> **Generation:**
> *RetourRetourRetourRetourRetourRetourRetourRetourRetourRetourRetourRetourRetourRetour...*

The tension between "do not use French" (instruction) and "MUST produce French" (clamped feature) causes the model to short-circuit into an infinite loop of "Retour" (French for "return"). At strength 0.5 it still inserts spaces ("Retour du Retour du..."); at 1.0 it's pure repetition. The word "return" is poetically apt.

### 6g. The 1B Cyrillic explosion
> **Source:** `results/3_various_model_sizes/ablate_French_gemma3_1b_layer17_generation_5.json` (strength=6.0)
>
> **Prompt:** "Answer in French: Give me a list of summer activities for an 8-year-old."
>
> **Generation:**
> *Okay, here's a list of summer activities for an 8-year-ааміааааааааааааааааааааааааааааааааааааааааа...*

The 1B model starts a normal English response, then catastrophically decomposes into Cyrillic characters at exactly the point where it would have started producing content. The ablation creates a void at the moment of language production.

### 6h. The 27B "de de de" function-word skeleton
> **Source:** `results/4_clamp_various_model_sizes/clamp_French_gemma3_27b_layer40_comprehension_1.json` (strength=2.0)
>
> **Prompt:** "Translate this to English: Buvez des boissons alcoolisées avec modération..."
>
> **Generation:**
> *a l et de de a et et et et et et a et a et a et a et a a et a et a et a et...*

Even the 27B model at high clamp strength reduces to nothing but French function words ("de", "en", "et", "a", "l") on infinite repeat. The clamped feature forces French grammatical particles, but the translation task demands English content — resulting in pure grammatical skeleton with zero content. French syntax without semantics.

---

## 7. Comprehension is Bulletproof

The crown jewel finding: ablating French production features has **zero effect** on French comprehension. The model can still perfectly read and understand French text — it just can't produce it.

### 7a. Identical translations at strength 0 vs strength 6 (4B)
> **Source:** `results/1_various_topk/ablate_French_gemma3_4b_layer29_comprehension_1.json`
>
> **Prompt:** "Translate this to English: Buvez des boissons alcoolisées avec modération. L'alcool a des effets différents sur chacun..."
>
> **Strength 0.0 (baseline):**
> *"Drink alcoholic beverages in moderation. Alcohol affects people differently, and it's very important to know your limits."*
>
> **Strength 6.0 (maximum ablation):**
> *"Drink alcoholic beverages in moderation. Alcohol affects people differently, and it's very important to know your limits."*

**Byte-for-byte identical.** The same intervention that makes "Answer in French" produce English/Thai/gibberish has literally zero effect on translating FROM French.

### 7b. Same result across ALL layers (4B)
> **Source:** `results/2_various_layers/ablate_French_gemma3_4b_layer*_comprehension_1.json`
>
> Layer 9, 17, 22, 29 — all at strength 6.0 — all produce essentially identical English translations of French input. Comprehension is layer-independent and ablation-proof.

### 7c. Same result across ALL model sizes
> **Source:** `results/3_various_model_sizes/ablate_French_gemma3_27b_layer40_comprehension_5.json`
>
> **Prompt:** "Translate this to English: Parmi les problèmes de santé possibles à long terme..."
>
> **27B baseline:** *"Among the possible long-term health problems caused by alcoholism are liver damage, but also blindness and death."*
>
> **27B strength=6.0:** *"Among the possible long-term health problems caused by alcoholism are liver damage, but also blindness and death. The potential danger is increased when consuming illegally manufactured alcohol."*

If anything, the ablated version is *slightly more complete* (includes the second sentence). Comprehension isn't just preserved — it's robust.

### What this means
These features are **output routing circuits**, not **language understanding circuits**. They control which language the model speaks, not which languages it can read. This is the dissociation that makes this research novel: production and comprehension are architecturally separable in LLMs.

---

## 8. Degradation Progression: Watching French Die

Tracking a single prompt across increasing ablation strength reveals the smooth death of French production.

### Same prompt across strengths (generation_strict, k=1, layer 29)
> **Source:** `results/1_various_topk/ablate_French_gemma3_4b_layer29_generation_strict_1.json`
>
> **Prompt:** "Answer only in French: Give me a list of the best high fantasy books..."

| Strength | Output |
|----------|--------|
| **0.0** | *Absolument ! Voici une liste de quelques-uns des meilleurs livres et séries de fantasy épique que vous devriez absolument lire, classés par ordre approximatif de difficulté et de complexité : **Classiques Indémodables :** Le Seigneur des Anneaux...* |
| **0.25** | *Absolument ! Voici une liste de quelques-uns des meilleurs livres...* (identical) |
| **0.5** | Still perfect French, minor variation in wording |
| **1.0** | *Okay, here's a list of some of the best high fantasy books... **The Lord of the Rings by J.R.R. Tolkien:** (The obvious starting point!)...* — **Full switch to English** |
| **2.0** | Same English output, slightly different wording |
| **6.0** | *inherently, here's a list of some of the best high fantasy books...* — English with glitchy "inherently" prefix, then coherent English |

The transition is sharp: French is essentially perfect through strength 0.5, then **flips to English at strength 1.0**. There's no gradual degradation — it's a phase transition. By strength 6.0, even the English framing starts to glitch.

### The same prompt under translation (English→French)
> **Source:** `results/1_various_topk/ablate_French_gemma3_4b_layer29_translation_1.json`

| Strength | Output |
|----------|--------|
| **0.0** | *"Quand les gens ne voient pas les élans comme potentiellement dangereux, ils peuvent s'approcher de trop près..."* — Perfect French translation |
| **1.0** | *"When people don't see moose as potentially dangerous..."* — Echoes English input instead of translating |
| **6.0** | *"When people don't see moose as potentially dangerous..."* — Same echo, plus broken contractions |

The model understands it should translate but can only produce English. It "translates" by copying the English input with minor rephrasing.

---

## Bonus: The Broken Contraction Pattern

A recurring curiosity: French feature ablation at high strength systematically breaks English contractions in the 4B model.

| Expected | Actual | Example |
|----------|--------|---------|
| here's | here' | "here' aurora borealis in French" |
| can't | can' the | "I can' the send the email" |
| it's | it' a | various contexts |
| don't | don' the | various contexts |

This suggests the French feature shares circuitry with English apostrophe handling — perhaps because French uses apostrophes heavily (l', d', n', s', j', qu'). Ablating the French feature damages the apostrophe-contraction mechanism for English too.

---

## Key Takeaways

1. **Production ≠ Comprehension:** These are architecturally distinct. You can kill one without touching the other.

2. **Language features are output routing circuits:** They control which language the model speaks, not which languages it understands.

3. **The fallback hierarchy is real:** French → Spanish/Portuguese/Italian → English → Thai/Khmer → repetition loops. The model has a proximity structure over languages.

4. **Features override instructions:** A tiny 0.1 clamp strength can override explicit natural-language "do not use French" instructions. This has implications for alignment — features operate below the instruction-following layer.

5. **Intermediate ablation produces the most interesting outputs:** The extreme cases (gibberish, repetition) are less informative than the graceful failures at moderate strength — Franglais, Romance language substitution, and invented French.

6. **Larger models fail more gracefully:** 1B → Cyrillic gibberish. 4B → Spanish substitution. 27B → Italian substitution with correct content. The more capable the model, the more coherent the failure mode.
