"""
Training examples for BootstrapFewShot compilation.

Each example covers a failure pattern observed during testing:
1. Adjacent-but-not-exact context
2. Short keyword-only query
3. Entity-type mismatch
4. Happy path with numbers
5. Multi-part question with partial coverage
"""

import dspy

trainset = [
    # --- 1. Adjacent-but-not-exact context ---
    dspy.Example(
        context=(
            "[Block ID: 1, Source: RBI_Master_Direction_CreditGuarantee.pdf]\n"
            "Banks are permitted to assign zero percent risk weight to the "
            "guaranteed portion of credit facilities extended under the Credit "
            "Guarantee Fund Scheme for Micro and Small Enterprises (CGTMSE). "
            "The guarantee cover is available for collateral-free credit "
            "facilities up to Rs 500 lakh extended by eligible institutions.\n\n"
            "[Block ID: 2, Source: RBI_CapitalAdequacy_Circular_2023.pdf]\n"
            "Capital funds are broadly classified as Tier 1 and Tier 2 capital. "
            "Common Equity Tier 1 must be at least 5.5 percent of risk-weighted "
            "assets. Banks must maintain a Capital Conservation Buffer of 2.5 "
            "percent, consisting of Common Equity Tier 1 capital."
        ),
        question="What is the minimum Capital to Risk (Weighted) Assets Ratio (CRAR) requirement under Basel III?",
        answer=(
            "The retrieved circulars cover capital fund classification and "
            "the Common Equity Tier 1 floor of 5.5 percent of risk-weighted "
            "assets [2], along with a Capital Conservation Buffer of 2.5 "
            "percent [2]. The specific overall CRAR minimum (typically the "
            "9 percent composite floor for Indian banks) is not stated in "
            "these particular context blocks. You would need to refer to "
            "the Master Circular on Basel III Capital Regulations for the "
            "exact composite figure."
        ),
    ).with_inputs("context", "question"),

    # --- 2. Short keyword-only query ---
    dspy.Example(
        context=(
            "[Block ID: 1, Source: RBI_KCC_Guidelines_2024.pdf]\n"
            "Kisan Credit Card (KCC) scheme is designed to provide adequate "
            "and timely credit support to farmers for their cultivation and "
            "other needs. Eligible farmers include owner cultivators, tenant "
            "farmers, oral lessees, and sharecroppers. The card covers "
            "post-harvest expenses, produce marketing loans, consumption "
            "requirements, working capital for allied activities such as "
            "dairy, fisheries, and poultry. The loan limit for marginal "
            "farmers is based on the scale of finance for the crop plus "
            "25 percent of the limit towards post-harvest and household needs.\n\n"
            "[Block ID: 2, Source: RBI_PSL_MasterDirection_2024.pdf]\n"
            "Loans under KCC scheme are classified under Priority Sector "
            "Lending. KCC loans for allied activities up to Rs 2 lakh are "
            "eligible for classification under agriculture."
        ),
        question="KCC",
        answer=(
            "The Kisan Credit Card scheme provides timely credit to farmers "
            "for cultivation, post-harvest expenses, produce marketing, and "
            "working capital for allied activities like dairy and fisheries [1]. "
            "Eligible borrowers include owner cultivators, tenant farmers, "
            "oral lessees, and sharecroppers [1]. KCC loans are classified "
            "under Priority Sector Lending, with allied-activity loans up "
            "to Rs 2 lakh qualifying under the agriculture sub-target [2]."
        ),
    ).with_inputs("context", "question"),

    # --- 3. Entity-type mismatch ---
    dspy.Example(
        context=(
            "[Block ID: 1, Source: RBI_UCB_Cybersecurity_Framework_2023.pdf]\n"
            "Urban Co-operative Banks (UCBs) shall implement a Board-approved "
            "cybersecurity policy. UCBs in Tier III and Tier IV must appoint "
            "a Chief Information Security Officer. All cyber incidents must "
            "be reported to CSITE within 6 hours of detection. UCBs shall "
            "conduct quarterly vulnerability assessments.\n\n"
            "[Block ID: 2, Source: RBI_IT_Framework_Banks_2023.pdf]\n"
            "All Scheduled Commercial Banks shall implement a comprehensive "
            "information security governance framework. Banks must establish "
            "a Security Operations Centre and report incidents to the RBI "
            "within 2 to 6 hours depending on severity classification."
        ),
        question="What are the cybersecurity requirements specifically for foreign banks operating in India?",
        answer=(
            "The retrieved context covers cybersecurity frameworks for Urban "
            "Co-operative Banks [1] and a general IT framework applicable to "
            "Scheduled Commercial Banks [2]. Foreign banks operating in India "
            "as Scheduled Commercial Banks would fall under the general SCB "
            "framework, which requires an information security governance "
            "framework, a Security Operations Centre, and incident reporting "
            "to RBI within 2 to 6 hours depending on severity [2]. The "
            "context does not contain a separate, foreign-bank-specific "
            "cybersecurity circular."
        ),
    ).with_inputs("context", "question"),

    # --- 4. Happy path with numbers ---
    dspy.Example(
        context=(
            "[Block ID: 1, Source: RBI_PSL_MasterDirection_2024.pdf]\n"
            "Domestic Scheduled Commercial Banks and foreign banks with 20 "
            "branches and above shall have a Priority Sector Lending target "
            "of 40 percent of Adjusted Net Bank Credit (ANBC) or Credit "
            "Equivalent Amount of Off-Balance Sheet Exposure (CEOBSE), "
            "whichever is higher. The agriculture sub-target is 18 percent "
            "of ANBC, with a sub-target of 10 percent for Small and Marginal "
            "Farmers within agriculture.\n\n"
            "[Block ID: 2, Source: RBI_PSL_MasterDirection_2024.pdf]\n"
            "Advances to weaker sections shall be 12 percent of ANBC or "
            "CEOBSE, whichever is higher. Micro enterprises under MSME "
            "shall have a sub-target of 7.5 percent of ANBC."
        ),
        question="What are the priority sector lending targets for domestic commercial banks?",
        answer=(
            "Domestic Scheduled Commercial Banks must direct 40 percent of "
            "ANBC or CEOBSE, whichever is higher, towards priority sector "
            "lending [1]. Within that, 18 percent of ANBC must go to "
            "agriculture, with 10 percent specifically earmarked for Small "
            "and Marginal Farmers [1]. The weaker sections sub-target is "
            "12 percent of ANBC [2], and micro enterprises under MSME carry "
            "a 7.5 percent sub-target [2]."
        ),
    ).with_inputs("context", "question"),

    # --- 5. Multi-part question with partial coverage ---
    dspy.Example(
        context=(
            "[Block ID: 1, Source: RBI_NPA_Classification_2024.pdf]\n"
            "A non-performing asset is a loan or advance where interest "
            "and/or instalment of principal remains overdue for a period of "
            "more than 90 days in respect of a term loan. For agricultural "
            "advances, the NPA classification norms are linked to crop "
            "seasons: a short-duration crop loan becomes NPA if the instalment "
            "of principal or interest remains overdue for two crop seasons, "
            "and a long-duration crop loan becomes NPA if overdue for one "
            "crop season.\n\n"
            "[Block ID: 2, Source: RBI_NPA_Classification_2024.pdf]\n"
            "Sub-standard assets are those that have remained NPA for a "
            "period less than or equal to 12 months. Doubtful assets are "
            "those that have remained in the sub-standard category for 12 "
            "months. Loss assets are those where loss has been identified "
            "but the amount has not been fully written off."
        ),
        question="What are the NPA classification norms for agricultural advances versus standard term loans, and what are the provisioning requirements for each category?",
        answer=(
            "For standard term loans, an account turns NPA when interest or "
            "principal remains overdue beyond 90 days [1]. Agricultural "
            "advances follow crop-season-based norms: short-duration crop "
            "loans become NPA after two crop seasons of overdue, and "
            "long-duration crop loans after one crop season [1].\n\n"
            "On sub-classification, sub-standard assets are those NPA for "
            "up to 12 months, doubtful assets have been sub-standard for "
            "12 months, and loss assets are those identified as losses but "
            "not fully written off [2]. The specific provisioning "
            "percentages for each category are not detailed in the "
            "retrieved context blocks."
        ),
    ).with_inputs("context", "question"),
]
