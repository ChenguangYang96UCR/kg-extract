from __future__ import annotations

import csv
import hashlib
import re
import unicodedata
from dataclasses import asdict, dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Iterable

DEFAULT_BASE_URI = "https://example.org/nsf/"
RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
SCHEMA = "https://schema.org/"
XSD = "http://www.w3.org/2001/XMLSchema#"


@dataclass(frozen=True, slots=True)
class Triple:
    subject: str
    predicate: str
    object: str
    object_type: str
    datatype: str = ""
    award_number: str = ""
    source_column: str = ""
    evidence: str = ""
    confidence: str = "1.0"
    extractor: str = "rules"


@dataclass(frozen=True, slots=True)
class ExtractionStats:
    awards: int
    triples: int
    entities: int
    skipped_rows: int


def clean_excel_literal(value: str | None) -> str:
    """Undo Excel's =\"value\" wrapper without changing ordinary values."""
    value = (value or "").strip()
    match = re.fullmatch(r'=\"(.*)\"', value, flags=re.DOTALL)
    return match.group(1).strip() if match else value


def normalize_date(value: str | None) -> str:
    value = clean_excel_literal(value)
    if not value:
        return ""
    for pattern in ("%m/%d/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, pattern).date().isoformat()
        except ValueError:
            pass
    return value


def normalize_money(value: str | None) -> str:
    value = clean_excel_literal(value)
    if not value:
        return ""
    normalized = value.replace("$", "").replace(",", "").strip()
    try:
        return format(Decimal(normalized), "f")
    except InvalidOperation:
        return value


def slug(value: str) -> str:
    ascii_value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    result = re.sub(r"[^a-z0-9]+", "-", ascii_value.lower()).strip("-")
    return result or "entity"


def entity_uri(base_uri: str, kind: str, label: str) -> str:
    normalized = " ".join(label.casefold().split())
    suffix = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:10]
    return f"{base_uri}{kind}/{slug(label)[:80]}-{suffix}"


def split_values(value: str | None) -> list[str]:
    return [part.strip() for part in (value or "").split(",") if part.strip()]


def parse_co_investigators(value: str | None) -> list[tuple[str, str]]:
    """Parse NSF's comma-separated `Name email` Co-PI field."""
    results: list[tuple[str, str]] = []
    for item in split_values(value):
        match = re.fullmatch(r"(.+?)\s+([^\s,]+@[^\s,]+)", item)
        if match:
            results.append((match.group(1).strip(), match.group(2).strip()))
        else:
            results.append((item, ""))
    return results


class AwardBuilder:
    def __init__(self, row: dict[str, str], base_uri: str, include_contact: bool) -> None:
        self.row = row
        self.base_uri = base_uri
        self.include_contact = include_contact
        self.number = clean_excel_literal(row.get("AwardNumber"))
        self.award = f"{base_uri}award/{self.number}"
        self.kg = f"{base_uri}vocab/"
        self.triples: list[Triple] = []

    def add(
        self,
        subject: str,
        predicate: str,
        obj: str,
        *,
        object_type: str = "literal",
        datatype: str = "",
        source: str = "",
        evidence: str | None = None,
        confidence: str = "1.0",
        extractor: str = "rules",
    ) -> None:
        if not obj:
            return
        self.triples.append(
            Triple(
                subject=subject,
                predicate=predicate,
                object=obj,
                object_type=object_type,
                datatype=datatype,
                award_number=self.number,
                source_column=source,
                evidence=obj if evidence is None and source else (evidence or ""),
                confidence=confidence,
                extractor=extractor,
            )
        )

    def add_entity(self, kind: str, label: str, relation: str, source: str) -> str:
        uri = entity_uri(self.base_uri, kind, label)
        self.add(self.award, self.kg + relation, uri, object_type="iri", source=source, evidence=label)
        self.add(uri, RDF_TYPE, self.kg + kind.title(), object_type="iri", source=source, evidence=label)
        self.add(uri, SCHEMA + "name", label, source=source)
        return uri

    def build(self) -> list[Triple]:
        row = self.row
        self.add(self.award, RDF_TYPE, SCHEMA + "MonetaryGrant", object_type="iri", source="AwardNumber")
        self.add(self.award, SCHEMA + "identifier", self.number, source="AwardNumber")
        self.add(self.award, SCHEMA + "name", clean_excel_literal(row.get("Title")), source="Title")
        self.add(self.award, SCHEMA + "description", row.get("Abstract", "").strip(), source="Abstract")

        self._add_literal("NSFOrganization", "nsfOrganizationCode")
        self._add_literal("State", "awardState")
        self._add_literal("AwardInstrument", "awardInstrument")
        self._add_literal("NSFDirectorate", "nsfDirectorateCode")
        self._add_literal("OrganizationState", "organizationState")
        self._add_literal("OrganizationCity", "organizationCity")
        self._add_literal("OrganizationZip", "organizationZip", cleaner=clean_excel_literal)

        self._add_date("StartDate", SCHEMA + "startDate")
        self._add_date("EndDate", SCHEMA + "endDate")
        self._add_date("LastAmendmentDate", self.kg + "lastAmendmentDate")
        self._add_money("AwardedAmountToDate", SCHEMA + "amount")
        self._add_money("ARRAAmount", self.kg + "arraAmount")

        pi = clean_excel_literal(row.get("PrincipalInvestigator"))
        if pi:
            pi_uri = self.add_entity("person", pi, "principalInvestigator", "PrincipalInvestigator")
            if self.include_contact:
                self.add(pi_uri, SCHEMA + "email", clean_excel_literal(row.get("PIEmailAddress")), source="PIEmailAddress")

        for name, email in parse_co_investigators(row.get("Co-PIName(s)")):
            person = self.add_entity("person", name, "coInvestigator", "Co-PIName(s)")
            if self.include_contact:
                self.add(person, SCHEMA + "email", email, source="Co-PIName(s)", evidence=email)

        manager = clean_excel_literal(row.get("ProgramManager"))
        if manager:
            self.add_entity("person", manager, "programManager", "ProgramManager")

        organization = clean_excel_literal(row.get("Organization"))
        if organization:
            org_uri = self.add_entity("organization", organization, "recipientOrganization", "Organization")
            if self.include_contact:
                self._add_org_contact(org_uri)

        for program in split_values(row.get("Program(s)")):
            self.add_entity("program", program, "belongsToProgram", "Program(s)")

        for code in split_values(row.get("ProgramElementCode(s)")):
            self.add(self.award, self.kg + "programElementCode", clean_excel_literal(code), source="ProgramElementCode(s)")
        for code in split_values(row.get("ProgramReferenceCode(s)")):
            self.add(self.award, self.kg + "programReferenceCode", clean_excel_literal(code), source="ProgramReferenceCode(s)")
        return self.triples

    def _add_literal(self, column: str, predicate: str, cleaner=clean_excel_literal) -> None:
        self.add(self.award, self.kg + predicate, cleaner(self.row.get(column)), source=column)

    def _add_date(self, column: str, predicate: str) -> None:
        value = normalize_date(self.row.get(column))
        datatype = XSD + "date" if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) else ""
        self.add(self.award, predicate, value, datatype=datatype, source=column)

    def _add_money(self, column: str, predicate: str) -> None:
        value = normalize_money(self.row.get(column))
        datatype = XSD + "decimal" if re.fullmatch(r"-?\d+(?:\.\d+)?", value) else ""
        self.add(self.award, predicate, value, datatype=datatype, source=column)

    def _add_org_contact(self, organization: str) -> None:
        row = self.row
        self.add(organization, SCHEMA + "telephone", clean_excel_literal(row.get("OrganizationPhone")), source="OrganizationPhone")
        parts = [
            clean_excel_literal(row.get("OrganizationStreet")),
            clean_excel_literal(row.get("OrganizationCity")),
            clean_excel_literal(row.get("OrganizationState")),
            clean_excel_literal(row.get("OrganizationZip")),
        ]
        address = ", ".join(part for part in parts if part)
        self.add(organization, SCHEMA + "address", address, source="OrganizationStreet", evidence=address)


def extract_awards(
    input_csv: str | Path,
    *,
    base_uri: str = DEFAULT_BASE_URI,
    include_contact: bool = False,
) -> tuple[list[Triple], ExtractionStats]:
    triples: list[Triple] = []
    awards = 0
    skipped = 0
    with Path(input_csv).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "AwardNumber" not in reader.fieldnames:
            raise ValueError("CSV must contain an AwardNumber column")
        for row in reader:
            if not clean_excel_literal(row.get("AwardNumber")):
                skipped += 1
                continue
            awards += 1
            triples.extend(AwardBuilder(row, base_uri, include_contact).build())

    entities = {triple.subject for triple in triples}
    entities.update(t.object for t in triples if t.object_type == "iri")
    stats = ExtractionStats(awards, len(triples), len(entities), skipped)
    return triples, stats


def write_csv(triples: Iterable[Triple], output_path: str | Path) -> None:
    fields = list(Triple.__dataclass_fields__)
    with Path(output_path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for triple in triples:
            writer.writerow(asdict(triple))


def _escape_literal(value: str) -> str:
    return (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
    )


def _as_ntriple(triple: Triple) -> str:
    subject = f"<{triple.subject}>"
    predicate = f"<{triple.predicate}>"
    if triple.object_type == "iri":
        obj = f"<{triple.object}>"
    else:
        obj = f'"{_escape_literal(triple.object)}"'
        if triple.datatype:
            obj += f"^^<{triple.datatype}>"
    return f"{subject} {predicate} {obj} ."


def write_ntriples(triples: Iterable[Triple], output_path: str | Path) -> None:
    with Path(output_path).open("w", encoding="utf-8", newline="\n") as handle:
        for triple in triples:
            handle.write(_as_ntriple(triple) + "\n")
