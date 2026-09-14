"""Render completed WRITE evidence; no experiments or model selection."""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.run_write_harm_structure_learnability import *


def table(df,columns=None):return df[columns].to_markdown(index=False,floatfmt='.4f') if columns else df.to_markdown(index=False,floatfmt='.4f')

def plot_lines(df,x,ys,name,ylabel):
    fig,ax=plt.subplots(figsize=(7,4))
    for label,column in ys:ax.plot(df[x],df[column],marker='o',label=label)
    ax.set(xlabel=x.replace('_',' '),ylabel=ylabel);ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(OUT/f'figures/{name}.png',dpi=180);plt.close(fig)

def local():
    verify();assert read_json(OUT/'learnability/complete.json')['passed'];d=pd.DataFrame(read_jsonl(OUT/'population/dense_write_state_manifest.jsonl'));s=read_json(OUT/'structure/complete.json');null=pd.read_csv(OUT/'structure/adjacent_persistence.csv');span=pd.read_csv(OUT/'structure/harmful_span_statistics.csv');neighbor=pd.read_csv(OUT/'structure/strong_flip_neighborhood.csv');relative=pd.read_csv(OUT/'structure/trigger_relative_summary.csv');cells=pd.read_csv(OUT/'structure/dataset_source_prevalence.csv');burden=pd.read_csv(OUT/'structure/sample_write_burden.csv');p=null[(null.cell=='ALL')&(null.metric=='harmful_persistence')].iloc[0];bq=burden.harmful_fraction.quantile([.5,.9,1]);spans=pd.read_csv(OUT/'structure/span_records.csv');length=spans[spans.sign=='harmful'].length.value_counts(normalize=True).sort_index()
    (OUT/'summaries/write_structure_summary.md').write_text(f'''# WRITE structure: complete dense census

All {s['states']:,} states / {s['uids']:,} UIDs / 1,385 image groups are included. Harmful/beneficial/zero counts: {s['sign_counts']}. Behavioral cohorts: {s['cohort_counts']}. Every UID contributes layer 27; the {int(((d.h_w==0)&(d.layer==27)).sum())} terminal WRITE interventions have zero downstream target effect. Zeros remain in the prospective primary denominator.

Harmful adjacent persistence is {p.observed:.4f} versus UID-shuffled mean {p.null_mean:.4f}, null 95% interval [{p.null_q025:.4f}, {p.null_q975:.4f}], enrichment {p.enrichment_ratio:.3f}x, empirical one-sided p={p.empirical_p_ge:.5f}. This is modest population-level organization under the specified within-UID layer shuffle, not a global predictable harmful regime. Source cells have unequal support and do not all exceed their null intervals.

{table(span)}

Harmful span proportions by length: {length.round(4).to_dict()}. The complete span records preserve longer tails. Strong-flip neighborhoods (offset 0 is selected on correctness, so its enrichment is not independent evidence):

{table(neighbor)}

Trigger-relative structure:

{table(relative)}

UID harmful-layer fraction median/90th percentile/maximum: {bq.iloc[0]:.3f}/{bq.iloc[1]:.3f}/{bq.iloc[2]:.3f}; {(burden.harmful_fraction>=.5).mean():.1%} of UIDs have at least half their layers harmful. This measures observed sample burden; it does not establish future-sample predictability.

Dataset/source prevalence:

{table(cells)}

WRITE harm shows modest adjacency and multi-layer spans, with many isolated signs and substantial source/sample variation. Structural organization alone does not establish a mechanism or learnable router.
''')
    f=pd.read_csv(OUT/'features/feature_distribution_summary.csv');e=pd.read_csv(OUT/'matched_mechanism/feature_effect_sizes.csv');e['excludes_zero']=(e.bootstrap_ci_low>0)|(e.bootstrap_ci_high<0);e['abs_effect']=e.standardized_effect_size.abs();effect=e[e.comparison=='harmful_vs_beneficial'].sort_values('abs_effect',ascending=False);mechanism=[]
    for family,desc in [('f1','magnitude'),('f2','direction'),('f3','token concentration'),('f4','visual diversity'),('f5','query alignment'),('f7','spatial concentration')]:
        g=effect[effect.feature.str.startswith(family+'_')];best=g.iloc[0];mechanism.append(dict(family=family,description=desc,feature=best.feature,effect=best.standardized_effect_size,raw_difference_ci_low=best.bootstrap_ci_low,raw_difference_ci_high=best.bootstrap_ci_high,features_ci_excludes_zero=int(g.excludes_zero.sum()),tested_features=len(g),pairs=int(best.pairs)))
    core=['f1_frobenius','f1_over_pre','f3_top1_share','f4_full_minus_off_effective_rank','f4_full_minus_off_pair_cosine','f5_full_minus_off_mean'];fm=f[f.feature.isin(core)];probes=pd.read_csv(OUT/'matched_mechanism/matched_probe_metrics.csv')
    (OUT/'summaries/write_mechanism_summary.md').write_text('''# WRITE mechanism diagnostic

Same-layer text parity passed on all 15,185 compact cached states, with full-text fresh replay on 32 UIDs /346 states. READ_ONLY visual states exactly equal the pre-WRITE states; both branch hashes match frozen parent artifacts. No target was regenerated. Features use all valid visual tokens, exact covariance spectrum, and validated image-grid coordinates. Optional attention-source F6 was omitted; this limits claims about attention-level causes.

Selected distribution summaries (FULL-minus-READ_ONLY updates):

'''+table(fm)+'''

Within exact nuisance-matched harmful versus beneficial correctness flips, the largest absolute standardized effect in each prespecified feature family is shown below. These are exploratory maxima across correlated features; unadjusted bootstrap intervals are not multiple-testing-corrected discovery claims. Image-group resampling uses both matched arms with shared multiplicities and separately normalized means.

'''+table(pd.DataFrame(mechanism))+'''

Full harmful-versus-stable-wrong/correct matches and all feature effects are retained in matched_mechanism. Positive effective-rank changes mean expansion, negative changes mean collapse; positive query-alignment change means stronger alignment with the fixed pre-query. Magnitude/direction/concentration contrasts are associations conditional on the exact matching support, not evidence that a geometric change caused answer harm.

Matched logistic/MLP probes:

'''+table(probes)+'''

No harmful-versus-beneficial feature interval excludes zero across all 85 frozen scalar features in this completed census. There is no stable feature family to designate as a robust mechanism; the largest exploratory standardized point contrast is spatial index spread, but its raw mean-difference interval includes zero. Matched probe AUROC is 0.4215 (linear) and 0.4132 (MLP), with 22 matched pairs. Only 40 beneficial flips exist before matching; strict matched support is consequently small. Probe support and unestimable folds are recorded separately. Feature effects do not by themselves justify a specific WRITE mechanism, a causal mediator, or deployment performance.
''')
    m=pd.read_csv(OUT/'learnability/all_local_metrics.csv');gate=read_json(OUT/'learnability/local_decision_gate.json');diff=pd.read_csv(OUT/'statistics/pairwise_model_differences.csv');columns=['population','feature_group','model','states','spearman','harmful_auroc','precision_at_0.1','recall_at_precision_0.9','harmful_flip_auroc'];columns=[c for c in columns if c in m]
    layers=pd.read_csv(OUT/'learnability/layer_breakdown.csv');source=pd.read_csv(OUT/'learnability/dataset_source_breakdown.csv');rank=pd.read_csv(OUT/'learnability/strong_flip_ranking.csv')
    (OUT/'summaries/write_learnability_summary.md').write_text('# WRITE local learnability: '+gate['category']+'\n\nComplete image-group-disjoint five-fold OOF estimates, mean of three raw-unit seed predictions per state. All models use the frozen capacity/optimizer/early-stop recipe, fold-local normalization and UID-balanced fit weights. Primary F_ALL/MLP was fixed prospectively. Generic-prestate and one-step delta baselines explicitly negate both old targets and raw predictions, then align exact states. Harmful means HW>0, with zero states included as nonharmful; legacy nonzero-only metrics are separately saved. Dense-W entries are actual refits.\n\n'+table(m[m.population!='matched'],columns)+'\n\nPaired 5,000-draw image-group differences:\n\n'+table(diff)+'\n\nThe gate requires material improvement over both baselines, Dense-W survival and useful high precision. Each component is recorded in local_decision_gate.json. A statistically nonzero but sub-threshold gain does not pass. All family ablations and fusion controls above remain secondary; no best-family substitution was made.\n\nEarly/middle/late breakdown:\n\n'+table(layers)+'\n\nDataset/source niches:\n\n'+table(source)+'\n\nSmall-cell highs are descriptive, not validated transfer. Strong correctness-flip ranking is measured separately from continuous harm and all-positive harm ranking:\n\n'+table(rank)+'\n\nUtility sign and correctness flips are distinct: 364 of 448 harmful correctness flips have positive HW, and 84 have negative HW; 37 of 40 beneficial flips have negative HW, and 3 have positive HW. The continuous surrogate and discrete behavior therefore require separate evaluations. This mismatch is descriptive and does not by itself explain model failures. The local study does not establish a useful router, benchmark improvement, savings, or impossibility of later/history-dependent information. '+('The frozen weak gate requires the authorized W3 propagation discriminator.' if gate['category']=='W-LOCAL-WEAK' else 'The positive gate stops W3 and permits the specified generalization checks.')+'\n')
    layer=pd.read_csv(OUT/'structure/absolute_layer_summary.csv');plot_lines(layer,'layer',[('harmful prevalence','harmful_prevalence'),('harmful-flip prevalence','harmful_flip_prevalence')],'write_harm_by_layer','Fraction of states');plot_lines(neighbor,'offset',[('harmful prevalence','harmful_prevalence')],'write_flip_neighborhood','Fraction harmful')
    fig,ax=plt.subplots(figsize=(6,4));ax.bar(length.index,length.values);ax.set(xlabel='Harmful span length',ylabel='Fraction of spans');fig.tight_layout();fig.savefig(OUT/'figures/write_harm_span_length.png',dpi=180);plt.close(fig)
    for feature,name in [('f1_frobenius','write_update_norm'),('f4_full_minus_off_effective_rank','write_visual_diversity_change'),('f5_full_minus_off_mean','write_query_alignment_change')]:
        g=f[(f.feature==feature)&(f.cohort!='ALL')];fig,ax=plt.subplots(figsize=(8,4));ax.bar(g.cohort,g['median']);ax.set(ylabel='Median '+feature);ax.tick_params(axis='x',rotation=15);fig.tight_layout();fig.savefig(OUT/f'figures/{name}.png',dpi=180);plt.close(fig)
    g=effect.head(15).iloc[::-1];fig,ax=plt.subplots(figsize=(9,6));ax.barh(g.feature,g.standardized_effect_size);ax.set(xlabel='Exploratory standardized matched effect');fig.tight_layout();fig.savefig(OUT/'figures/write_feature_effect_sizes.png',dpi=180);plt.close(fig)
    g=m[(m.population=='full')&(m.model=='mlp')];fig,ax=plt.subplots(figsize=(9,4));ax.bar(g.feature_group,g.spearman);ax.set(ylabel='OOF Spearman');ax.tick_params(axis='x',rotation=30);fig.tight_layout();fig.savefig(OUT/'figures/write_learnability_comparison.png',dpi=180);plt.close(fig)
    print('Local summaries and eight figures complete',flush=True)


def propagation():
    verify();assert read_json(OUT/'propagation/complete.json')['passed'];m=pd.read_csv(OUT/'propagation/horizon_metrics.csv');support=pd.read_csv(OUT/'propagation/horizon_support.csv');growth=pd.read_csv(OUT/'propagation/effect_growth_statistics.csv');decision=read_json(OUT/'propagation/propagation_decision.json');diff=pd.read_csv(OUT/'statistics/propagation_pairwise_differences.csv');columns=['horizon','support','population','condition','states','spearman','harmful_auroc','precision_at_0.1','recall_at_precision_0.9','harmful_flip_auroc'];common=m[m.support=='common']
    records=pd.DataFrame(read_jsonl(OUT/'propagation/propagated_features.jsonl'));records=records[records.layer<=19];common_growth=[]
    assert records.state_id.nunique()==6044 and len(records)==6044*5
    for family in ['write_sign','cohort']:
        for (h,label),g in records.groupby(['horizon',family]):common_growth.append(dict(horizon=h,family=family,cohort=label,states=len(g),text_l2_mean=g.text_l2.mean(),text_l2_median=g.text_l2.median(),visual_l2_mean=g.visual_l2.mean(),visual_l2_median=g.visual_l2.median(),text_relative_l2_mean=g.text_relative_l2.mean(),visual_relative_l2_mean=g.visual_relative_l2.mean(),nonzero_text_fraction=(g.text_l2>0).mean()))
    common_growth=pd.DataFrame(common_growth);common_growth.to_csv(OUT/'propagation/effect_growth_common_support.csv',index=False)
    first=records[records.text_l2>0].groupby('state_id').horizon.min();first_rows=records[records.horizon==0][['state_id','cohort']].copy();first_rows['first_measured_text_divergence']=first_rows.state_id.map(first).fillna(-1).astype(int);first_rows.to_csv(OUT/'propagation/first_text_divergence_common_support.csv',index=False)
    first_counts=first_rows.first_measured_text_divergence.value_counts().sort_index().to_dict()
    (OUT/'summaries/write_propagation_summary.md').write_text('# WRITE propagation: '+decision['category']+'\n\nWRITE H0 is immediately after intervention layer l; Hk includes k subsequent FULL layers through l+k. READ remains ON. Both branches use identical FULL continuation; no later routing/search or target regeneration. ON raw states reuse exact parent reached-layer FULL references; OFF raw compact text/all-visual states are retained and hashed. Fresh smoke verifies all ON horizons, H0 cache parity, repeated OFF, prestate identities and action traces.\n\nNative support:\n\n'+table(support)+'\n\nPrimary emergence uses the exact H8-common support (l<=19); native-support results remain descriptive. Native and common H8 are identical and share fitted predictions. Text is the last control token, visual is pooled by its mean; raw visual norms use all tokens. This is a fixed pooled-representation diagnostic, not a test of every token-level architecture.\n\nPrimary effect growth on exact H8-common support:\n\n'+table(common_growth)+'\n\nFirst measured text-divergence horizon over these 6,044 states: '+str(first_counts)+' (-1 means none through H8).\n\nDescriptive native-support effect growth:\n\n'+table(growth)+'\n\nExact common-support OOF metrics, with Dense-W models independently refitted:\n\n'+table(common,columns)+'\n\nNative-support metrics:\n\n'+table(m[m.support=='native'],columns)+'\n\nCommon-support paired bootstrap differences:\n\n'+table(diff)+'\n\nThe prospective primary is combined delta/MLP. Material emergence, advantage over both single branches, Dense-W survival and high-precision checks are recorded in propagation_decision.json. Secondary ordered-pair and text/visual ablations do not replace the primary after outcomes. Positive text divergence establishes downstream influence of the WRITE intervention under this fixed continuation; it does not establish predictable answer harm or identify a sufficient causal mediator. A weak result is bounded by the target, support, horizons and representation/capacity family.\n')
    for metric,name in [('spearman','write_horizon_spearman'),('harmful_auroc','write_horizon_harmful_auroc')]:
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        for ax,pop in zip(axes,['full','dense_w']):
            for condition,g in common[common.population==pop].groupby('condition'):ax.plot(g.horizon,g[metric],marker='o',label=condition)
            ax.set(title=pop,xlabel='WRITE horizon',ylabel=metric);ax.grid(alpha=.2)
        axes[1].legend(fontsize=8);fig.tight_layout();fig.savefig(OUT/f'figures/{name}.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for label,g in common_growth[common_growth.family=='write_sign'].groupby('cohort'):
        axes[0].plot(g.horizon,g.text_relative_l2_mean,marker='o',label=label);axes[1].plot(g.horizon,g.visual_relative_l2_mean,marker='o',label=label)
    for ax,stream in zip(axes,['Text/control — H8 common support','All visual — H8 common support']):ax.set(title=stream,xlabel='WRITE horizon',ylabel='Mean relative L2');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'figures/write_text_vs_visual_divergence_by_horizon.png',dpi=180);plt.close(fig)
    print('Propagation summary and figures complete',flush=True)

def comparison():
    """Descriptive comparison of completed studies, with actual elapsed layers."""
    m=pd.read_csv(OUT/'propagation/horizon_metrics.csv')
    w=m[(m.support=='common')&(m.population=='full')&(m.condition=='delta')].sort_values('horizon')
    read_path=ROOT/'analysis/read_harm_short_horizon_propagation/summaries/read_short_horizon_propagation_summary.md'
    # Rounded values transcribed from the completed READ summary, not new fits.
    r=pd.DataFrame({'subsequent_layers':[0,1,3,7],'spearman':[.0756,.0688,.0646,.1097],'harmful_auroc':[.5316,.5287,.5314,.5483]})
    rows=pd.concat([r.assign(operation='READ',states=6916,source=str(read_path.relative_to(ROOT))),w.rename(columns={'horizon':'subsequent_layers'})[['subsequent_layers','spearman','harmful_auroc']].assign(operation='WRITE',states=6044,source='analysis/write_harm_structure_learnability/propagation/horizon_metrics.csv')],ignore_index=True)
    rows.to_csv(OUT/'summaries/read_vs_write_curve_data.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(11,4.8))
    for ax,metric,title in zip(axes,['spearman','harmful_auroc'],['OOF Spearman','Harmful AUROC']):
        for operation,g in rows.groupby('operation'):
            ax.plot(g.subsequent_layers,g[metric],marker='o',label=f'{operation}: {g.states.iloc[0]:,} states')
        ax.set(xlabel='FULL layers after the intervention layer',ylabel=title,xticks=range(9));ax.grid(alpha=.2);ax.legend()
    axes[1].axhline(.5,color='gray',ls=':',lw=1)
    fig.suptitle('Completed pooled-delta curves on each study’s own common support')
    fig.text(.5,.015,'Descriptive comparison: different targets and cohorts; no paired READ–WRITE superiority test.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.05,1,.94]);fig.savefig(OUT/'figures/read_vs_write_summary.png',dpi=180);plt.close(fig)
    print('READ versus WRITE comparison figure complete',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['local','propagation','comparison']);a=p.parse_args();{'local':local,'propagation':propagation,'comparison':comparison}[a.command]()
