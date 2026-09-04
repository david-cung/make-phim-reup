# Subtitle / Translation Architecture Rules

## CRITICAL RULES

These rules are mandatory and must be treated as architectural constraints.

1. Never hard-code Vietnamese pronouns from gender.

   Forbidden examples:

   ```text
   female -> em
   male -> anh
   female -> chị
   male -> ông
   ```

2. Speaker identity must remain stable across the entire video whenever available evidence indicates the same speaker.

3. Vietnamese pronouns must be resolved from:

   ```text
   speaker
   + listener
   + relationship
   + dialogue context
   + established pronoun history
   ```

   Gender alone is never sufficient.

4. Character and dialogue context must persist for the entire video processing session.

   Do not recreate or reset character context for every subtitle segment.

5. Do not fix missing speaker, gender, listener, relationship, or pronoun context only by adding more instructions to the translation prompt.

   Fix the correct architectural layer first.

The intended processing hierarchy is:

```text
video/audio analysis
        ↓
speaker identity
        ↓
character identity/context
        ↓
listener inference
        ↓
relationship inference
        ↓
pronoun resolution
        ↓
contextual translation
        ↓
consistency validation
        ↓
subtitle / TTS
```

Prompt engineering may complement this architecture but must not replace it.

---

# Product Goal

The application is a video-first automatic subtitle and dubbing system.

The final end-user workflow should be:

```text
Upload video
    ↓
Automatic analysis
    ↓
Automatic transcription
    ↓
Automatic speaker/character understanding
    ↓
Automatic contextual translation
    ↓
Automatic consistency validation
    ↓
Subtitle / dubbing output
```

The user should NOT be required to manually provide:

- speaker gender
- character names
- number of speakers
- speaker mappings
- character mappings
- relationships
- Vietnamese pronouns
- dialogue context
- who is speaking to whom

The system should infer as much as possible automatically from:

- audio
- video
- dialogue
- subtitle/source text
- surrounding scenes
- previously accumulated context

---

# General Engineering Principle

Do not treat subtitle translation as isolated sentence translation.

This application must progressively understand the video.

Information discovered earlier or later in the video may be used to improve:

- speaker identity
- character identity
- listener identity
- gender hints
- relationships
- pronoun mappings
- translation consistency

Prefer evidence accumulation over one-time guesses.

---

# Inspect Before Modifying

Before implementing changes related to:

- subtitle generation
- transcription
- speaker detection
- gender inference
- character identity
- translation
- Vietnamese pronouns
- TTS voice assignment

first inspect the existing implementation.

Required workflow:

1. Trace the current pipeline.
2. Identify existing abstractions.
3. Identify existing speaker/context state.
4. Find the actual root cause.
5. Reuse existing functionality where appropriate.
6. Modify the smallest correct architectural layer.
7. Add or update tests.
8. Run relevant tests.

Do not create duplicate systems when equivalent functionality already exists.

Do not assume architecture that has not been verified in the repository.

---

# Stable Speaker Identity

Every detected voice should use a stable identity when possible.

Preferred IDs:

```text
SPEAKER_00
SPEAKER_01
SPEAKER_02
```

The same speaker should not receive a new logical identity for every subtitle segment.

Speaker identity should persist across the video whenever speaker diarization or other evidence indicates the same person is speaking.

Conceptually support:

```text
SpeakerProfile
```

with information such as:

```text
speaker_id
voice_reference
voice_embedding
gender_hint
gender_confidence
character_id
dialogue_history
```

Do not derive stable identity only from subtitle ordering.

---

# Character Identity

Speaker identity and character identity are related but are not necessarily the same abstraction.

The architecture should support:

```text
SPEAKER_02
        ↓
CHARACTER_05
```

A character may later be associated with:

- voice identity
- face track
- active speaker evidence
- dialogue history
- names
- relationships

Use stable character IDs such as:

```text
CHARACTER_00
CHARACTER_01
CHARACTER_02
```

Character names are optional.

The application must work even when the character's real name is unknown.

---

# Character Context Must Be Video-Scoped

Character state must persist for the lifetime of the current video-processing job.

Do not reset this state for every:

- subtitle line
- translation batch
- scene
- LLM request

The architecture should conceptually support a:

```text
CharacterContextStore
```

Example information:

```text
character_id
associated_speaker_ids
gender_hint
gender_confidence
age_hint
role_hint
dialogue_history
relationships
pronoun_mappings
context_confidence
```

This state should evolve as more evidence becomes available.

---

# Gender Is Evidence, Not Pronoun Selection

Gender must never directly determine Vietnamese forms of address.

Gender inference should support:

```text
male
female
unknown
uncertain
```

Prefer confidence-aware values:

```json
{
  "gender": "female",
  "confidence": 0.71
}
```

Low-confidence gender inference must not become an irreversible fact.

Later evidence must be allowed to update it.

Do not assume:

```text
female speaker = em
male speaker = anh
```

because this is linguistically incorrect.

---

# Evidence-Based Gender Inference

Where available, gender-related inference may combine evidence from:

```text
voice characteristics
+
visual appearance
+
dialogue content
+
how other characters address this character
+
relationship evidence
+
previous dialogue
```

No single signal should automatically override all others unless confidence is sufficiently strong.

Do not fake gender certainty.

---

# Pronouns Are Relationship-Based

A character must NOT have one global Vietnamese pronoun.

Incorrect model:

```text
CHARACTER_01.pronoun = "em"
```

Correct conceptual model:

```text
CHARACTER_01 -> CHARACTER_02

self = em
other = anh
```

while the same character may use:

```text
CHARACTER_01 -> CHARACTER_03

self = con
other = bố
```

Therefore pronoun state must depend on:

```text
speaker_character
+
listener_character
+
relationship
+
conversation context
```

Suggested conceptual structure:

```text
PronounMapping
```

containing:

```text
speaker_character_id
listener_character_id
self_pronoun
listener_pronoun
relationship
confidence
evidence
last_updated
```

---

# Preserve Established Pronouns

When a speaker/listener pair develops a reliable pronoun mapping, reuse it.

Example:

```text
CHARACTER_01 -> CHARACTER_02

self = em
other = anh
confidence = high
```

Subsequent translations should strongly prefer that mapping.

Do not allow arbitrary changes such as:

```text
em / anh
↓
chị / em
↓
tôi / cô
↓
anh / chị
```

without meaningful contextual evidence.

Established pronouns are soft constraints, not permanent hard-coded values.

They may change if:

- relationship changes
- social situation changes
- dialogue explicitly changes formality
- story context provides contradictory evidence

---

# Speaker and Listener Are Different Concepts

Always distinguish:

```text
who is speaking
```

from:

```text
who is being addressed
```

Do not infer Vietnamese self-pronouns only from the speaker.

Example:

A female character may use:

```text
em
chị
mẹ
con
cháu
tôi
tao
mình
```

depending on the listener and situation.

Therefore listener inference is a first-class part of contextual translation.

---

# Listener / Addressee Inference

For every dialogue segment, attempt to determine the likely listener when enough evidence exists.

Possible evidence includes:

- previous speaker
- next speaker
- turn-taking
- scene participants
- names
- vocatives
- source-language address terms
- gaze or active-speaker information when available
- relationship history
- surrounding dialogue

Support:

```text
listener = unknown
```

Do not force a listener when evidence is insufficient.

---

# Relationship Inference

Relationships may include concepts such as:

```text
romantic
parent_child
siblings
friends
coworkers
manager_employee
teacher_student
customer_staff
strangers
unknown
```

These are conceptual categories, not a required fixed enum unless the current implementation benefits from one.

Relationships should be confidence-aware.

Example:

```json
{
  "relationship": "siblings",
  "confidence": 0.58
}
```

Do not turn weak guesses into permanent facts.

---

# Relationships Can Evolve

The system must allow later dialogue to update previous assumptions.

Example:

Initially:

```text
CHARACTER_01 -> CHARACTER_04
relationship = unknown
```

Later:

```text
"Dad, wait for me."
```

This may provide evidence that:

```text
CHARACTER_01 -> CHARACTER_04
relationship = parent_child
```

Update context when justified.

Do not create architecture where early assumptions are impossible to revise.

---

# Translation Must Use Dialogue Context

Do not translate subtitle lines as isolated sentences when contextual data exists.

Translation should receive a bounded context window.

Useful context may include:

```text
current speaker
likely listener
previous dialogue
nearby next dialogue
character context
gender hints
relationship
established pronouns
scene information
confidence values
```

Example conceptual translation context:

```text
Speaker:
CHARACTER_01

Listener:
CHARACTER_02

Speaker gender hint:
female, confidence 0.91

Listener gender hint:
male, confidence 0.86

Relationship:
romantic, confidence 0.84

Established mapping:
self = em
other = anh

Previous dialogue:
...

Current source:
Why didn't you tell me?
```

Expected translation should preserve established context:

```text
Sao anh không nói cho em biết?
```

Do not arbitrarily produce unrelated pronouns.

---

# Bounded Context

Do not send the entire movie transcript to the LLM for every subtitle segment.

Use:

- bounded dialogue windows
- reusable character state
- reusable relationship state
- scene summaries where appropriate
- cached inference

Avoid unnecessary:

```text
O(N²)
```

processing.

Do not repeatedly recompute expensive information when it can be safely reused.

---

# Translation Priority

When translating dialogue into Vietnamese, use this priority:

1. Preserve source meaning.
2. Preserve speaker identity.
3. Preserve listener identity.
4. Preserve known relationship.
5. Preserve established forms of address.
6. Preserve emotional tone and register.
7. Produce natural Vietnamese dialogue.
8. Avoid literal word-for-word translation when it damages meaning.

---

# Unknown Context

When context is uncertain:

Do:

```text
unknown
uncertain
low confidence
```

when appropriate.

Do not invent:

- gender
- age
- family relationship
- romantic relationship
- social hierarchy

just to fill metadata.

When Vietnamese permits it, prefer natural wording that avoids unnecessary unsupported gender assumptions.

---

# Pronoun Consistency Validator

Translation generation and consistency validation should be separate concerns.

Where supported by the current architecture, add a consistency validation stage after translation.

Example:

Established mapping:

```text
CHARACTER_01 -> CHARACTER_02

self = em
other = anh
```

Previous lines:

```text
em / anh
em / anh
em / anh
```

New translation:

```text
anh / chị
```

This must be treated as suspicious.

Re-evaluate using:

```text
original source
speaker
listener
relationship
surrounding dialogue
established pronoun mapping
confidence
```

Repair only affected subtitle segments when possible.

Do not unnecessarily retranslate the entire video.

---

# Global Consistency Pass

After translation of the full video, allow a lightweight global consistency pass.

Example:

```text
CHARACTER_01 self-reference statistics:

em: 85
tôi: 6
anh: 1
```

The single:

```text
anh
```

may indicate a translation error.

Use global frequency only for anomaly detection.

Do NOT automatically replace text based only on frequency.

Always re-evaluate suspicious lines using context and source dialogue.

---

# Context Must Not Leak Between Character Pairs

Pronoun mappings must remain pair-specific.

Example:

```text
CHARACTER_A -> CHARACTER_B
self = em
other = anh
```

must not accidentally affect:

```text
CHARACTER_A -> CHARACTER_C
```

or:

```text
CHARACTER_D -> CHARACTER_B
```

unless context explicitly connects them.

Avoid shared mutable state that causes unrelated character relationships to contaminate each other.

---

# Multimodal Architecture

The long-term application must be multimodal.

Audio analysis may include:

```text
speech recognition
voice activity detection
speaker diarization
voice embeddings
speech timing
```

Video analysis may include:

```text
face detection
face tracking
face clustering
active speaker detection
scene detection
visual character identification
```

The architecture should eventually support:

```text
voice speaker
+
face track
+
active speaker evidence
+
dialogue context
=
character identity
```

Do not tightly couple translation logic to one specific diarization, ASR, or computer-vision model.

Use clean interfaces where practical.

---

# Active Speaker Mapping

Future architecture should allow mappings such as:

```text
SPEAKER_02
+
FACE_TRACK_07
+
lip/audio synchronization
=
CHARACTER_03
```

This allows the application to determine which visible character corresponds to a detected voice.

Do not fake this mapping when active-speaker analysis has not been implemented.

---

# Do Not Fake Missing Features

If the current repository does not support:

- speaker diarization
- face detection
- face tracking
- active speaker detection
- visual character identification

do not fabricate results.

Instead:

1. preserve current functionality
2. use currently available evidence
3. create appropriate extension points if needed
4. keep unknown values where appropriate
5. document remaining limitations

---

# TTS Character Persistence

TTS voice selection should eventually use stable character identity.

The same character should not randomly receive different synthesized voices across the same video unless explicitly justified.

Conceptually:

```text
CHARACTER_01
        ↓
TTS_VOICE_03
```

should persist across that video.

Do not couple this directly to Vietnamese pronoun selection.

Voice identity and linguistic pronoun resolution are separate concerns.

---

# Preserve Existing Functionality

Changes related to this architecture must not unnecessarily break:

- video upload
- file handling
- audio extraction
- transcription
- subtitle segmentation
- subtitle timing
- translation
- subtitle export
- TTS
- video rendering
- configuration
- existing APIs
- existing previous phases

Prefer incremental changes.

Do not rewrite working previous phases unless there is a demonstrated architectural requirement.

---

# Backwards Compatibility

When introducing new context metadata:

- provide safe defaults
- support existing stored jobs if applicable
- avoid breaking existing serialization unnecessarily
- migrate schemas carefully if persistence exists

Unknown context should degrade gracefully.

---

# Logging and Diagnostics

Provide useful diagnostics for this pipeline.

For a subtitle segment, developers should be able to inspect information conceptually similar to:

```text
segment_id
speaker_id
character_id
listener_id
speaker_gender_hint
speaker_gender_confidence
listener_gender_hint
relationship
relationship_confidence
pronoun_mapping
pronoun_confidence
source_text
translated_text
validator_result
```

Follow the project's existing logging conventions.

Do not expose excessive technical diagnostics in the normal end-user interface.

---

# Performance

Avoid unnecessary repeated model execution.

Cache or reuse when appropriate:

- speaker identity
- character state
- relationship state
- pronoun mapping
- scene context
- model embeddings

Do not run expensive audio/video models multiple times over identical data unless required.

Do not send excessive context to translation APIs.

Keep the desktop application responsive where practical.

---

# Testing Requirements

Any implementation change affecting:

- speaker identity
- gender inference
- character context
- listener inference
- relationship inference
- Vietnamese pronouns
- contextual translation
- consistency validation

must include or update relevant tests.

At minimum cover these regression scenarios.

## Female speaker consistency

Female Character A speaking to Male Character B.

If context establishes:

```text
A = em
B = anh
```

multiple consecutive lines must not randomly flip A into:

```text
anh
chị
```

without contextual reason.

## Male speaker consistency

Male Character A speaking to Female Character B.

Pronouns must remain stable when context is unchanged.

## Parent / child

Gender alone must not produce:

```text
anh / em
```

when dialogue establishes:

```text
bố / con
mẹ / con
```

## Unknown relationship

The system must preserve uncertainty rather than inventing a relationship.

## Relationship learned later

Initial relationship:

```text
unknown
```

Later dialogue establishes the relationship.

CharacterContextStore should update correctly.

## Pronoun outlier

Twenty consistent lines followed by one inconsistent line.

The validator should identify the suspicious line.

## Multiple speakers

Pronoun state from one speaker/listener pair must not leak into another pair.

## Character identity persistence

The same speaker across distant scenes should preserve identity when evidence supports it.

## Low-confidence inference

Low-confidence gender or relationship inference must remain revisable.

---

# Implementation Rule for Bug Fixes

When receiving a bug such as:

```text
female character translated as male
male character translated as female
wrong anh/em
wrong chị/em
pronouns changing between lines
```

DO NOT immediately patch the translation prompt.

First determine which layer failed:

```text
speaker detection?
character identity?
listener inference?
relationship inference?
pronoun state?
context propagation?
translation prompt?
consistency validator?
```

Fix the earliest incorrect layer.

Only modify the translation prompt when the architecture already provides the correct context and the translation model is failing to follow it.

---

# No Narrow Hard-Coded Fixes

Avoid fixes tied to:

- one movie
- one subtitle file
- one character name
- one specific sentence
- one specific language example
- one speaker ID observed in a test video

The application must remain generic.

Prefer reusable inference and validation logic.

---

# Codex Working Rules for This Area

When Codex works on speaker / character / subtitle / translation logic:

1. Read this AGENTS.md section first.
2. Inspect the existing implementation before editing.
3. Trace the complete relevant data flow.
4. Identify the root cause.
5. Reuse existing abstractions where possible.
6. Do not solve architectural context problems only with prompt text.
7. Avoid hard-coded gender-to-pronoun rules.
8. Preserve backwards compatibility.
9. Add tests for the reported regression.
10. Run relevant tests.
11. Fix regressions introduced by the change.
12. Do not modify unrelated features.
13. Clearly report remaining limitations.

---

# Definition of Done

A speaker/gender/pronoun improvement is not considered complete merely because one example subtitle now looks correct.

The implementation should demonstrate that:

- speaker identity is stable where possible
- character context persists
- uncertainty is supported
- pronouns are relationship-based
- translation receives available context
- established pronouns are preserved
- suspicious pronoun flips can be detected
- unrelated character context does not leak
- current video/subtitle/TTS functionality still works
- tests cover the regression
- no simplistic female/male -> Vietnamese pronoun mapping was introduced
