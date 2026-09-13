from cora.ingest import build_query, parse_pubmed_xml
from cora import config

XML = """<?xml version="1.0"?>
<PubmedArticleSet>
 <PubmedArticle>
  <MedlineCitation><PMID>111</PMID>
   <Article>
    <Journal><Title>J Test</Title><JournalIssue><PubDate><Year>2020</Year></PubDate></JournalIssue></Journal>
    <ArticleTitle>A titled <i>italic</i> study</ArticleTitle>
    <Abstract>
      <AbstractText Label="BACKGROUND">First part.</AbstractText>
      <AbstractText Label="RESULTS">Second part with <sup>2</sup> markup.</AbstractText>
    </Abstract>
   </Article>
  </MedlineCitation>
  <PubmedData><History>
    <PubMedPubDate PubStatus="entrez"><Year>2020</Year><Month>3</Month><Day>7</Day></PubMedPubDate>
  </History></PubmedData>
 </PubmedArticle>
 <PubmedArticle>
  <MedlineCitation><PMID>222</PMID>
   <Article>
    <Journal><Title>J Old</Title><JournalIssue><PubDate><MedlineDate>1998 Jan-Feb</MedlineDate></PubDate></JournalIssue></Journal>
    <ArticleTitle>No abstract here</ArticleTitle>
   </Article>
  </MedlineCitation>
 </PubmedArticle>
 <PubmedArticle>
  <MedlineCitation><PMID>333</PMID>
   <Article>
    <Journal><Title>J Old</Title><JournalIssue><PubDate><MedlineDate>1998 Jan-Feb</MedlineDate></PubDate></JournalIssue></Journal>
    <ArticleTitle>Medline date fallback</ArticleTitle>
    <Abstract><AbstractText>Plain abstract text.</AbstractText></Abstract>
   </Article>
  </MedlineCitation>
 </PubmedArticle>
</PubmedArticleSet>"""


def test_parse_joins_labels_and_inline_markup():
    docs = parse_pubmed_xml(XML)
    by = {d["pmid"]: d for d in docs}
    assert "222" not in by  # no abstract -> dropped
    assert by["111"]["title"] == "A titled italic study"
    assert by["111"]["abstract"] == "BACKGROUND: First part. RESULTS: Second part with 2 markup."
    assert by["111"]["pub_year"] == 2020 and by["111"]["entrez_date"] == "2020-03-07"
    assert by["111"]["journal"] == "J Test"
    assert by["333"]["pub_year"] == 1998 and by["333"]["entrez_date"] is None


FLAGGED = """<?xml version="1.0"?>
<PubmedArticleSet>
 <PubmedArticle>
  <MedlineCitation><PMID>444</PMID>
   <Article>
    <Journal><Title>J</Title><JournalIssue><PubDate><Year>2015</Year></PubDate></JournalIssue></Journal>
    <ArticleTitle>Flagged paper</ArticleTitle>
    <Abstract><AbstractText>Some abstract text here.</AbstractText></Abstract>
    <PublicationTypeList><PublicationType>Journal Article</PublicationType><PublicationType>Retracted Publication</PublicationType></PublicationTypeList>
   </Article>
   <CommentsCorrectionsList>
     <CommentsCorrections RefType="RetractionIn"><RefSource>J 2020</RefSource><PMID>555</PMID></CommentsCorrections>
     <CommentsCorrections RefType="CommentIn"><PMID>556</PMID></CommentsCorrections>
     <CommentsCorrections RefType="ErratumIn"><PMID>557</PMID></CommentsCorrections>
   </CommentsCorrectionsList>
  </MedlineCitation>
 </PubmedArticle>
 <PubmedArticle>
  <MedlineCitation><PMID>445</PMID>
   <Article><ArticleTitle>No abstract but flagged</ArticleTitle></Article>
   <CommentsCorrectionsList><CommentsCorrections RefType="ExpressionOfConcernIn"><PMID>558</PMID></CommentsCorrections></CommentsCorrectionsList>
  </MedlineCitation>
 </PubmedArticle>
</PubmedArticleSet>"""


def test_parse_retraction_and_erratum_flags():
    docs = {d["pmid"]: d for d in parse_pubmed_xml(FLAGGED)}
    assert docs["444"]["flags"] == ["pubtype:Retracted Publication", "RetractionIn:555", "ErratumIn:557"]
    assert "445" not in docs  # no abstract, dropped by default
    kept = {d["pmid"]: d for d in parse_pubmed_xml(FLAGGED, keep_empty=True)}
    assert kept["445"]["flags"] == ["ExpressionOfConcernIn:558"] and kept["445"]["abstract"] == ""


def test_queries_use_binomials_not_common_names():
    q = build_query(config.SPECIES["ocean_quahog"])
    assert "Arctica islandica" in q and "quahog" not in q.lower()
    q = build_query(config.SPECIES["rockfish"])
    assert "Sebastes" in q and "rockfish" not in q.lower()
    assert "longevity" in q
