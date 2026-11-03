import pytest
from src.nlp_ei.authentic import extract_sections,SecClient,evaluate_labels


def test_extractor_skips_table_of_contents_and_marks_missing():
    html="<p>Item 1A.</p><p>Item 1B.</p><h2>Item 1A. Risk Factors</h2>"+"Risk language. "*200+"<h2>Item 1B. Comments</h2>"
    r=extract_sections(html)
    assert r["risk_factors"]["characters"]>1000
    assert r["mda"]["status"]=="missing"


def test_contact_required(tmp_path):
    with pytest.raises(ValueError):SecClient("placeholder",tmp_path)


def test_labels_cannot_be_missing():
    with pytest.raises(ValueError):evaluate_labels([{"annotator_a":"","annotator_b":""}])


def test_confusion_and_disagreement():
    r=evaluate_labels([{"annotator_a":"negative","annotator_b":"neutral","adjudicated_label":"negative","predicted_label":"negative"},{"annotator_a":"positive","annotator_b":"positive","adjudicated_label":"positive","predicted_label":"neutral"}])
    assert r["human_disagreement_rate"]==.5
    assert r["confusion_matrix"]==[[1,0,0],[0,0,0],[0,1,0]]


def test_inline_cross_reference_does_not_expand_section():
    html="<p>For details see <span>Item 1A. Risk Factors</span> on page 10.</p>"+"Business discussion. "*100
    html+="<h2>Item 1A. Risk Factors</h2>"+"Actual risk text. "*100+"<h2>Item 1B. Unresolved Staff Comments</h2>"
    r=extract_sections(html)["risk_factors"]
    assert "Business discussion" not in r["text"]
    assert r["text"].startswith("Item 1A. Risk Factors")


def test_incorporated_sections_explicitly_missing():
    html="<h2>Item 7. Management discussion</h2><p>Information is incorporated by reference to the annual report.</p><h2>Item 7A. Market risk</h2>"
    r=extract_sections(html)["mda"]
    assert r["status"]=="missing"
    assert r["reason"]=="incorporated_by_reference_or_not_present"
