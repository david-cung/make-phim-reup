use std::collections::{BTreeMap, BTreeSet};
use std::io;
use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::subtitles::{SubtitleDoc, SubtitleSegment};

pub const PRONOUNS_RELATIVE: &str = "metadata/pronouns.json";
const PRONOUNS_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct CharacterProfile {
    pub id: String,
    pub display_name: String,
    #[serde(default)]
    pub speaker_ids: Vec<String>,
    #[serde(default)]
    pub notes: String,
    #[serde(default = "unknown")]
    pub gender_presentation: String,
    #[serde(default = "unknown")]
    pub age_group: String,
    #[serde(default)]
    pub default_self_reference: String,
    #[serde(default)]
    pub default_neutral_address: String,
    #[serde(default)]
    pub user_defined: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct RelationshipRule {
    pub from_character_id: String,
    pub to_character_id: String,
    #[serde(default)]
    pub relationship_type: String,
    #[serde(default)]
    pub self_reference: String,
    #[serde(default)]
    pub address_term: String,
    #[serde(default)]
    pub confidence: f32,
    #[serde(default = "manual_source")]
    pub source: String,
    #[serde(default)]
    pub user_defined: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct SegmentPronounFlag {
    pub segment_id: u32,
    #[serde(default)]
    pub flags: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speaker_character_id: Option<String>,
    #[serde(default)]
    pub addressee_character_ids: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rule_key: Option<String>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct PronounContextDoc {
    pub version: u32,
    #[serde(default)]
    pub characters: Vec<CharacterProfile>,
    #[serde(default)]
    pub relationships: Vec<RelationshipRule>,
    #[serde(default)]
    pub review_flags: Vec<SegmentPronounFlag>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone)]
pub struct SegmentPronounContext {
    pub speaker_character_id: Option<String>,
    pub addressee_character_ids: Vec<String>,
    pub rule: Option<RelationshipRule>,
    pub flags: Vec<String>,
}

pub struct PronounCacheFile;

impl PronounCacheFile {
    pub fn load(project_root: &Path) -> io::Result<PronounContextDoc> {
        let path = manifest_path(project_root);
        match std::fs::read_to_string(&path) {
            Ok(text) => {
                let mut doc: PronounContextDoc = serde_json::from_str(&text)
                    .map_err(|e| io::Error::other(format!("invalid pronouns.json: {e}")))?;
                normalise_doc(&mut doc);
                Ok(doc)
            }
            Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(PronounContextDoc::empty()),
            Err(err) => Err(err),
        }
    }

    pub fn save(project_root: &Path, doc: &PronounContextDoc) -> io::Result<PathBuf> {
        let mut doc = doc.clone();
        doc.updated_at = Utc::now();
        normalise_doc(&mut doc);
        let path = manifest_path(project_root);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let tmp = path.with_extension("json.tmp");
        let text =
            serde_json::to_string_pretty(&doc).map_err(|e| io::Error::other(e.to_string()))?;
        std::fs::write(&tmp, text)?;
        std::fs::rename(&tmp, &path)?;
        Ok(path)
    }
}

impl PronounContextDoc {
    pub fn empty() -> Self {
        Self {
            version: PRONOUNS_SCHEMA_VERSION,
            characters: Vec::new(),
            relationships: Vec::new(),
            review_flags: Vec::new(),
            updated_at: Utc::now(),
        }
    }
}

pub fn manifest_path(project_root: &Path) -> PathBuf {
    project_root.join(PRONOUNS_RELATIVE)
}

pub fn segment_contexts(
    doc: &PronounContextDoc,
    subtitles: Option<&SubtitleDoc>,
) -> BTreeMap<u32, SegmentPronounContext> {
    let mut out = BTreeMap::new();
    let Some(subtitles) = subtitles else {
        return out;
    };
    let index = build_index(doc);
    for (idx, segment) in subtitles.segments.iter().enumerate() {
        out.insert(
            segment.id,
            resolve_segment_context(doc, subtitles, &index, idx, segment),
        );
    }
    out
}

pub fn context_to_wire(ctx: &SegmentPronounContext, doc: &PronounContextDoc) -> Value {
    let speaker = ctx
        .speaker_character_id
        .as_deref()
        .and_then(|id| character_by_id(doc, id))
        .map(character_to_wire);
    let addressees = ctx
        .addressee_character_ids
        .iter()
        .filter_map(|id| character_by_id(doc, id))
        .map(character_to_wire)
        .collect::<Vec<_>>();
    let rule = ctx.rule.as_ref().map(|rule| {
        json!({
            "fromCharacterId": rule.from_character_id,
            "toCharacterId": rule.to_character_id,
            "relationshipType": rule.relationship_type,
            "selfReference": rule.self_reference,
            "addressTerm": rule.address_term,
            "confidence": rule.confidence,
            "source": rule.source,
            "userDefined": rule.user_defined,
        })
    });
    json!({
        "speaker": speaker,
        "addressees": addressees,
        "relationshipRule": rule,
        "reviewFlags": ctx.flags,
    })
}

pub fn upsert_review_flag(
    doc: &mut PronounContextDoc,
    segment_id: u32,
    flags: Vec<String>,
    ctx: Option<&SegmentPronounContext>,
) {
    let mut flags = flags
        .into_iter()
        .filter(|s| !s.trim().is_empty())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    if flags.is_empty() {
        doc.review_flags.retain(|f| f.segment_id != segment_id);
        doc.updated_at = Utc::now();
        return;
    }
    flags.sort();
    let rule_key = ctx.and_then(|c| c.rule.as_ref().map(relationship_key));
    let speaker_character_id = ctx.and_then(|c| c.speaker_character_id.clone());
    let addressee_character_ids = ctx
        .map(|c| c.addressee_character_ids.clone())
        .unwrap_or_default();
    if let Some(existing) = doc
        .review_flags
        .iter_mut()
        .find(|f| f.segment_id == segment_id)
    {
        existing.flags = flags;
        existing.speaker_character_id = speaker_character_id;
        existing.addressee_character_ids = addressee_character_ids;
        existing.rule_key = rule_key;
        existing.updated_at = Utc::now();
    } else {
        doc.review_flags.push(SegmentPronounFlag {
            segment_id,
            flags,
            speaker_character_id,
            addressee_character_ids,
            rule_key,
            updated_at: Utc::now(),
        });
    }
    doc.updated_at = Utc::now();
}

pub fn changed_relationship_keys(
    before: &PronounContextDoc,
    after: &PronounContextDoc,
) -> BTreeSet<String> {
    let before_map = before
        .relationships
        .iter()
        .map(|r| (relationship_key(r), comparable_rule(r)))
        .collect::<BTreeMap<_, _>>();
    let after_map = after
        .relationships
        .iter()
        .map(|r| (relationship_key(r), comparable_rule(r)))
        .collect::<BTreeMap<_, _>>();
    let mut keys = BTreeSet::new();
    for key in before_map.keys().chain(after_map.keys()) {
        if before_map.get(key) != after_map.get(key) {
            keys.insert(key.clone());
        }
    }
    keys
}

pub fn changed_speaker_ids(
    before: &PronounContextDoc,
    after: &PronounContextDoc,
) -> BTreeSet<String> {
    let before_map = speaker_map(before);
    let after_map = speaker_map(after);
    let mut out = BTreeSet::new();
    for key in before_map.keys().chain(after_map.keys()) {
        if before_map.get(key) != after_map.get(key) {
            out.insert(key.clone());
        }
    }
    out
}

pub fn relationship_key(rule: &RelationshipRule) -> String {
    format!("{}->{}", rule.from_character_id, rule.to_character_id)
}

pub fn obvious_pronoun_flags(text: &str, rule: &RelationshipRule) -> Vec<String> {
    let expected = [&rule.self_reference, &rule.address_term]
        .into_iter()
        .map(|s| s.trim().to_lowercase())
        .filter(|s| !s.is_empty())
        .collect::<BTreeSet<_>>();
    if expected.is_empty() {
        return Vec::new();
    }
    let known = [
        "anh", "chị", "em", "cô", "chú", "bác", "ông", "bà", "con", "cháu", "tôi", "mình",
    ];
    let words = text
        .split(|c: char| !c.is_alphabetic())
        .map(|s| s.trim().to_lowercase())
        .filter(|s| !s.is_empty())
        .collect::<BTreeSet<_>>();
    let conflict = known
        .iter()
        .any(|p| words.contains(*p) && !expected.contains(*p));
    if conflict {
        vec![
            "POSSIBLE_PRONOUN_INCONSISTENCY".into(),
            "USER_REVIEW_RECOMMENDED".into(),
        ]
    } else {
        Vec::new()
    }
}

fn resolve_segment_context(
    doc: &PronounContextDoc,
    subtitles: &SubtitleDoc,
    index: &PronounIndex,
    segment_index: usize,
    segment: &SubtitleSegment,
) -> SegmentPronounContext {
    let speaker_character_id = segment
        .speaker
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .and_then(|speaker| index.speaker_to_character.get(speaker).cloned());
    let addressee_character_ids = resolve_addressees(
        doc,
        subtitles,
        index,
        segment_index,
        segment,
        speaker_character_id.as_deref(),
    );
    let rule = if addressee_character_ids.len() == 1 {
        speaker_character_id
            .as_deref()
            .and_then(|from| relationship_between(doc, from, &addressee_character_ids[0]).cloned())
    } else {
        None
    };
    let mut flags = Vec::new();
    if speaker_character_id.is_none() {
        flags.push("UNKNOWN_SPEAKER".into());
    }
    if addressee_character_ids.is_empty() {
        flags.push("UNKNOWN_ADDRESSEE".into());
    }
    if addressee_character_ids.len() > 1 {
        flags.push("AMBIGUOUS_RELATIONSHIP".into());
    }
    if speaker_character_id.is_some() && !addressee_character_ids.is_empty() && rule.is_none() {
        flags.push("PRONOUN_INFERENCE_USED".into());
        flags.push("USER_REVIEW_RECOMMENDED".into());
    }
    SegmentPronounContext {
        speaker_character_id,
        addressee_character_ids,
        rule,
        flags,
    }
}

fn resolve_addressees(
    doc: &PronounContextDoc,
    subtitles: &SubtitleDoc,
    index: &PronounIndex,
    segment_index: usize,
    segment: &SubtitleSegment,
    speaker_character_id: Option<&str>,
) -> Vec<String> {
    let text = format!(
        "{}\n{}\n{}",
        segment.source_text, segment.translated_text, segment.dubbing_text
    )
    .to_lowercase();
    let explicit = doc
        .characters
        .iter()
        .filter(|c| Some(c.id.as_str()) != speaker_character_id)
        .filter(|c| {
            let name = c.display_name.trim().to_lowercase();
            !name.is_empty() && text.contains(&name)
        })
        .map(|c| c.id.clone())
        .collect::<BTreeSet<_>>();
    if !explicit.is_empty() {
        return explicit.into_iter().collect();
    }

    let mut nearby = BTreeSet::new();
    for prev in subtitles.segments[..segment_index].iter().rev().take(3) {
        collect_other_speaker(index, prev, speaker_character_id, &mut nearby);
        if nearby.len() > 1 {
            break;
        }
    }
    if nearby.len() <= 1 {
        for next in subtitles.segments[segment_index + 1..].iter().take(2) {
            collect_other_speaker(index, next, speaker_character_id, &mut nearby);
            if nearby.len() > 1 {
                break;
            }
        }
    }
    nearby.into_iter().collect()
}

fn collect_other_speaker(
    index: &PronounIndex,
    segment: &SubtitleSegment,
    speaker_character_id: Option<&str>,
    out: &mut BTreeSet<String>,
) {
    let Some(speaker) = segment
        .speaker
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
    else {
        return;
    };
    let Some(character_id) = index.speaker_to_character.get(speaker) else {
        return;
    };
    if Some(character_id.as_str()) != speaker_character_id {
        out.insert(character_id.clone());
    }
}

fn relationship_between<'a>(
    doc: &'a PronounContextDoc,
    from: &str,
    to: &str,
) -> Option<&'a RelationshipRule> {
    doc.relationships
        .iter()
        .find(|r| r.from_character_id == from && r.to_character_id == to)
}

fn character_by_id<'a>(doc: &'a PronounContextDoc, id: &str) -> Option<&'a CharacterProfile> {
    doc.characters.iter().find(|c| c.id == id)
}

fn character_to_wire(c: &CharacterProfile) -> Value {
    json!({
        "id": c.id,
        "displayName": c.display_name,
        "genderPresentation": c.gender_presentation,
        "ageGroup": c.age_group,
        "defaultSelfReference": c.default_self_reference,
        "defaultNeutralAddress": c.default_neutral_address,
        "userDefined": c.user_defined,
    })
}

#[derive(Debug)]
struct PronounIndex {
    speaker_to_character: BTreeMap<String, String>,
}

fn build_index(doc: &PronounContextDoc) -> PronounIndex {
    PronounIndex {
        speaker_to_character: speaker_map(doc),
    }
}

fn speaker_map(doc: &PronounContextDoc) -> BTreeMap<String, String> {
    let mut map = BTreeMap::new();
    for character in &doc.characters {
        for speaker in &character.speaker_ids {
            let speaker = speaker.trim();
            if !speaker.is_empty() {
                map.insert(speaker.to_string(), character.id.clone());
            }
        }
    }
    map
}

fn comparable_rule(rule: &RelationshipRule) -> (String, String, String) {
    (
        rule.relationship_type.trim().to_string(),
        rule.self_reference.trim().to_string(),
        rule.address_term.trim().to_string(),
    )
}

fn normalise_doc(doc: &mut PronounContextDoc) {
    doc.version = PRONOUNS_SCHEMA_VERSION;
    for character in &mut doc.characters {
        character.id = stable_id(&character.id, &character.display_name);
        character.display_name = character.display_name.trim().to_string();
        character.speaker_ids = character
            .speaker_ids
            .iter()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        if character.gender_presentation.trim().is_empty() {
            character.gender_presentation = "unknown".into();
        }
        if character.age_group.trim().is_empty() {
            character.age_group = "unknown".into();
        }
    }
    doc.characters.retain(|c| !c.display_name.is_empty());
    for relationship in &mut doc.relationships {
        relationship.from_character_id = relationship.from_character_id.trim().to_string();
        relationship.to_character_id = relationship.to_character_id.trim().to_string();
        relationship.self_reference = relationship.self_reference.trim().to_string();
        relationship.address_term = relationship.address_term.trim().to_string();
        if relationship.source.trim().is_empty() {
            relationship.source = manual_source();
        }
        relationship.confidence = relationship.confidence.clamp(0.0, 1.0);
    }
    let character_ids = doc
        .characters
        .iter()
        .map(|c| c.id.clone())
        .collect::<BTreeSet<_>>();
    doc.relationships.retain(|r| {
        r.from_character_id != r.to_character_id
            && character_ids.contains(&r.from_character_id)
            && character_ids.contains(&r.to_character_id)
    });
    doc.review_flags.sort_by_key(|f| f.segment_id);
}

fn stable_id(id: &str, display_name: &str) -> String {
    let raw = if id.trim().is_empty() {
        display_name
    } else {
        id
    };
    let mut out = String::from("character_");
    for ch in raw.trim().to_lowercase().chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch);
        } else if ch.is_whitespace() || ch == '-' || ch == '_' {
            out.push('_');
        }
    }
    while out.contains("__") {
        out = out.replace("__", "_");
    }
    if out == "character_" {
        format!("character_{}", Utc::now().timestamp_millis())
    } else {
        out.trim_end_matches('_').to_string()
    }
}

fn unknown() -> String {
    "unknown".into()
}

fn manual_source() -> String {
    "manual".into()
}
