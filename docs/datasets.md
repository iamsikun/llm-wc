# LLM Datasets and Benchmarks

**DialogStudio** is a collection of dialog datasets gathered by Salesforce. It contains [87 datasets](https://github.com/salesforce/DialogStudio/blob/main/Dataset_Stats.csv) and categorizes them into six categories: knowledge grounded dialogues, natural language understanding, open domain dialogues, task oriented dialogues, dialogue summarization, and conversational recommendation dislogs. 
* Paper: [DialogStudio: Towards Richest and Most Diverse Unified Dataset Collection for Conversational AI](https://arxiv.org/pdf/2307.10172)
* Link: [GitHub](https://github.com/salesforce/DialogStudio), [HuggingFace](https://huggingface.co/datasets/Salesforce/dialogstudio)


## Task-Oriented Dialog (TOD) Datasets
These are multi‑turn, goal‑driven conversations where the agent must gather slots, call APIs, and finish a task (book, order, schedule, etc.). They include explicit goals, schemas/slots, and often success metrics (e.g., Inform, Success, or “correct API + parameters”).

**MultiWOZ 2.4**. 
* Introduction: A refined, multi‑domain human–human dialog benchmark with cleaned annotations relative to 2.1. It covers typical assistant domains (hotel, restaurant, attraction, taxi, train, hospital, police) and is widely used for end‑to‑end evaluation (NLU → state tracking → action → NLG).  
* Summary statistics: Roughly 10k dialogs across 7 domains, average 8–10 turns per dialog, with belief state annotations and dialog acts. Standard metrics include *Inform* (did the system provide the requested entity) and *Success* (did it satisfy all constraints/booking). Good for reporting end‑to‑end task completion and state‑tracking robustness.
* Paper: [MultiWOZ 2.4: A Multi-Domain Task-Oriented Dialogue Dataset with Essential Annotation Corrections to Improve State Tracking Evaluation](https://arxiv.org/abs/2104.00773)
* Link: [GitHub](https://github.com/smartyfh/MultiWOZ2.4)
* Example (annotations = dialogue state metadata per domain on assistant turn):
```json
{
  "turns": [
    {
      "speaker": "USER",
      "text": "am looking for a place to to stay that has cheap price range it should be in a type of hotel"
    },
    {
      "speaker": "ASSISTANT",
      "text": "Okay, do you have a specific area you want to stay in?",
      "annotations": {
        "hotel": {
          "book": {
            "booked": [],
            "stay": "",
            "day": "",
            "people": ""
          },
          "semi": {
            "type": "",
            "parking": "",
            "pricerange": "",
            "internet": "",
            "name": "",
            "stars": "",
            "area": ""
          }
        }
      }
    }
  ]
}
```


**Schema-Guided Dialogue (SGD) and SGD-eXtended (SGD-X)**. 
* Introduction: Large, multi‑domain dataset designed to test generalization to unseen services via schema descriptions (intents, slots, slot types). Emphasizes API invocation correctness and cross‑service transfer.  Over 20k annotated multi-domain, task-oriented human-assistant conversations across 20 domains; SGD-X adds five paraphrased schema variants to test robustness to linguistic variation.
* Summary statistics: Around 18–20k dialogs spanning ~20 domains and ~45 services, with schemas, intents, slots, and API call labels. Evaluation focuses on slot accuracy, dialog state tracking, and API/argument correctness, which serves as a direct proxy for completion with back‑end actions.
* Paper: [Towards Scalable Multi-Domain Conversational Agents: The Schema-Guided Dialogue Dataset](https://arxiv.org/abs/1909.05855)
* Link: [GitHub](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue#scheme-representation%5D)
* Example (annotations = frame actions and dialogue state for each turn):
```json
{
  "turns": [
    {
      "speaker": "USER",
      "utterance": "I am feeling hungry so I would like to find a place to eat.",
      "annotations": [
        {
          "actions": [
            {
              "act": "INFORM_INTENT",
              "slot": "intent",
              "values": [
                "FindRestaurants"
              ]
            }
          ],
          "service": "Restaurants_1",
          "state": {
            "active_intent": "FindRestaurants",
            "requested_slots": [],
            "slot_values": {}
          }
        }
      ]
    },
    {
      "speaker": "SYSTEM",
      "utterance": "Do you have a specific which you want the eating place to be located at?",
      "annotations": [
        {
          "actions": [
            {
              "act": "REQUEST",
              "slot": "city",
              "values": []
            }
          ],
          "service": "Restaurants_1"
        }
      ]
    }
  ]
}
```

**Taskmaster (TM-1/2/3/4)**
* Introduction: Task dialogs (both spoken and written) across numerous consumer tasks provided
* Producer: Google
* Link: [GitHub](https://github.com/google-research-datasets/Taskmaster)
* Taskmaster 1
    * Paper: [Taskmaster-1: Toward a Realistic and Diverse Dialog Dataset](https://arxiv.org/abs/1909.05358)
    * Summary statistics: 13,215 dialogs with 5,507 spoken and 7,708 written. 
* Taskmaster 2
    * Summary statistics: 17,829 dialogs (all spoken) in seven domains: restaurants (3,276), food ordering (1,050), movies (3,047), hotels (2,355), flights (2,481), music (1602), sports (3,478). 
* Taskmaster 3
    * Summary statistics: 23,789 movie ticketing dialogs. Movie ticketing means conversations where the customer's goal is to purchase tickets after deciding on theater, time, movie name, number of tickets, and date, or opt out of the transaction.
* Taskmaster 4
    * Paper: [Parameter Efficient Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2403.10704)
    * Summary statistics: 6,500 coffee ordering dialogs and 3,000 reward examples
* Example (from Taskmaster 1):
```json
{
        "index": 0,
        "speaker": "USER",
        "text": "Hi, I'm looking to book a table for Korean food."
      },
      {
        "index": 1,
        "speaker": "ASSISTANT",
        "text": "Ok, what area are you thinking about?"
      },
      {
        "index": 2,
        "speaker": "USER",
        "text": "Somewhere in Southern NYC, maybe the East Village?",
        "segments": [
          {
            "start_index": 13,
            "end_index": 49,
            "text": "Southern NYC, maybe the East Village",
            "annotations": [
              {
                "name": "restaurant_reservation.location.restaurant.accept"
              }
            ]
          },
          {
            "start_index": 13,
            "end_index": 25,
            "text": "Southern NYC",
            "annotations": [
              {
                "name": "restaurant_reservation.location.restaurant.accept"
              }
            ]
          }
        ]
      }
```

**AirDialogue**
* Introduction: Constrained flight‑booking dialogs where success requires satisfying itinerary constraints (dates, cities, connections, budgets). 
* Summary statistics: Approximately 400k dialogs with explicit goals and exact‑match success based on fulfilling constraints and producing a valid booking.
* Paper: [AirDialogue: An Environment for Goal-Oriented Dialogue Research](https://aclanthology.org/D18-1419/)
* Link: [GitHub](https://github.com/google/airdialogue)
* Example (annotations = `intent` goal constraints and `action` target state):
```json
{
  "dialogue": [
    "customer: Hello.",
    "agent: Hello. How may I help you?"
  ],
  "intent": {
    "return_month": "Sept",
    "return_day": "13",
    "max_price": 5000,
    "departure_airport": "IAD",
    "max_connections": 1,
    "departure_day": "11",
    "goal": "change",
    "departure_month": "Sept",
    "name": "Edward Hall",
    "return_airport": "ATL"
  },
  "action": {
    "status": "no_reservation",
    "name": "Edward hall",
    "flight": []
  }
}
```

## Knowledge-Grounded Support Dialog Datasets

Knowledge Grounded Support Dialogs are grounded in external documents or multi‑document collections (policies, FAQs, manuals). Good systems must retrieve the right spans and produce policy‑consistent answers. These sets are ideal for measuring policy‑grounded completion and are more realistic for enterprise support where errors cause escalations or compliance issues.

**Doc2Dial**  
* Introduction: Goal-oriented, document-grounded information-seeking dialogues collected from public-sector websites.  
* Summary statistics: ~4–5k dialogs, each linked to document passages with turn‑level grounding annotations. Use this set to quantify policy‑consistent completion: count a turn as successful only if the answer and its cited span are both correct.
* Provider: IBM
* Paper: [doc2dial: A Goal-Oriented Document-Grounded Dialogue Dataset](https://arxiv.org/abs/2011.06623)  
* Link: [Website](https://doc2dial.github.io/), [HuggingFace](https://huggingface.co/datasets/IBM/doc2dial)  
* Example (assistant cites the specific document span used):
```json
{
  "dialog_id": "d2d-00123",
  "turns": [
    {"role": "user", "text": "When is the FAFSA deadline for next fall?"},
    {"role": "assistant",
     "text": "For most states, the priority deadline is in early spring. Your state’s deadline is listed on the FAFSA form.",
     "grounding": {
       "doc_id": "studentaid.gov#fafsa-deadlines",
       "spans": [{"start_char": 210, "end_char": 290, "section": "Deadlines"}]
     }}
  ]
}
```

**MultiDoc2Dial**  
* Introduction: An extension to multi‑document grounding, requiring the system to retrieve and integrate multiple sources to answer a single question. Ideal for measuring retrieval + reasoning + compliance. It is a closer proxy to enterprise help‑center bots.
* Summary statistics: A few thousand grounded turns with multi‑doc references and span IDs. 
* Provider: IBM
* Paper: [MultiDoc2Dial: Modeling Dialogues Grounded in Multiple Documents](https://aclanthology.org/2021.emnlp-main.498/)  
* Link: [Website](https://doc2dial.github.io/multidoc2dial/), [GitHub](https://github.com/IBM/multidoc2dial)  
* Example (assistant turn grounded in multiple docs):
```json
{
  "dialog_id": "mdd-04567",
  "turn_id": 6,
  "context_docs": ["ssa.gov#benefits-eligibility", "ssa.gov#apply-online"],
  "user": "Do I qualify for retirement benefits and how do I apply online?",
  "assistant": {
    "text": "You’re eligible if you have 40 credits; to apply, sign in and submit the application online.",
    "grounding": [
      {"doc_id": "ssa.gov#benefits-eligibility", "span_id": "p3"},
      {"doc_id": "ssa.gov#apply-online", "span_id": "steps-1-3"}
    ]
  }
}
```

## Summarization Datasets

These contain conversational transcripts with human summaries. They are useful to mine closure/next‑step language*and to train an outcome detector (resolved vs escalated vs follow‑up). 

**DialogSum**  
* Introduction: A large-scale dialogue summarization dataset, each dialog has the corresponding manually labeled summaries and topics.
* Summary statistics: 13,460 dialogs. Labels = abstractive summary + topic
* Paper: [DialogSum: A Real-Life Scenario Dialogue Summarization Dataset](https://arxiv.org/abs/2105.06762)  
* Link: [GitHub](https://github.com/cylnlp/dialogsum)  
* Example:
```json
{
  "id": "ds_13460",
  "dialogue": [
    "#Person1#: I'd like to cancel my gym membership.",
    "#Person2#: I can help. Do you have your member ID?",
    "#Person1#: Yes, it's 100234...",
    "#Person2#: Done. You'll receive a confirmation email."
  ],
  "summary": "Member requests cancellation; agent verifies ID and completes the cancellation.",
  "topic": "Customer Service"
}
```

## Intent, Triage, and Out-of-Scope (OOS) Datasets

Single‑turn user queries labeled with fine‑grained intents and explicit OOS examples. These are the front‑door of any production system: they determine what should be auto‑escalated immediately.

**BANKING77**  
* Introduction: Single‑turn banking queries annotated with 77 fine‑grained intents
* Summary statistics: 13k examples balanced across 77 intents
* Link: [HuggingFace](https://huggingface.co/datasets/PolyAI/banking77)  
* Example:
```json
{
    "text": "I lost my card. How do I freeze it?", 
    "label": "card_not_received"
}
```

**CLINC150 / OOS-eval**  
* Introduction: Multi‑domain single‑turn queries with 150 intents and a dedicated Out‑of‑Scope evaluation set. 
* Summary statistics: ~22–23k in‑domain utterances across 10 domains / 150 intents, plus OOS test examples. 
* Paper: [An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction](https://arxiv.org/abs/1909.02027)  
* Link: [GitHub (oos-eval)](https://github.com/clinc/oos-eval), [TFDS](https://www.tensorflow.org/datasets/catalog/clinc_oos)  
* Example:
```json
{
    "text": "book me a flight to denver tomorrow morning", 
    "label": "travel_book"
}, 
{
    "text": "how to bake sourdough?", 
    "label": "oos"
}
```
