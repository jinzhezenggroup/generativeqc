#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include "scf/occupied_projection.hpp"

namespace {
using namespace generativeqc::scf::projection;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

void close(double actual, double expected, double tolerance = 2e-12) {
  require(std::abs(actual - expected) <= tolerance, "independent occupied projection gate failed");
}

template <class Operation>
void rejected(Operation operation) {
  bool refused = false;
  try {
    operation();
  } catch (const Rejected&) {
    refused = true;
  }
  require(refused, "invalid occupied projection was accepted");
}

void retained_metric_purity_regression() {
  // Seeded ambient orthonormal occupied space in a rotated metric with
  // eigenvalues from 1 through 1e-8. This is a pure six-electron density:
  // independent relative ||D S D - 2 D|| / ||D|| is 1.76e-9. The inherited
  // scalar metric stopping tolerance gave populations 1.999999869 and
  // 2.000000180 and incorrectly rejected the default public 1e-7 purity gate.
  // A self-projection must preserve this density; no coefficient gauge is fixed.
  const std::array<double, 49> metric{{
      0.10897864075922167,   0.12328738309282247,   0.097452074469936131,  -0.17106197816482369,
      0.020340741862348216,  0.13888895579193616,   -0.14212987114032397,  0.12328738309282247,
      0.15476478030608273,   0.10802362174775998,   -0.20442735847938831,  0.027566268287782119,
      0.18384541882085675,   -0.15675309739702856,  0.097452074469936131,  0.10802362174775998,
      0.087580775339421338,  -0.15112897774612005,  0.017660033057217351,  0.11993484598908515,
      -0.12825907397806091,  -0.17106197816482369,  -0.20442735847938831,  -0.15112897774612005,
      0.27694531893172636,   -0.034849870669314831, -0.23811301354328471,  0.21881316822144872,
      0.020340741862348216,  0.027566268287782119,  0.017660033057217351,  -0.034849870669314831,
      0.0053229478080235098, 0.033353248457052224,  -0.026042238957297705, 0.13888895579193616,
      0.18384541882085675,   0.11993484598908515,   -0.23811301354328471,  0.033353248457052224,
      0.22556924649878665,   -0.171888066028661,    -0.14212987114032397,  -0.15675309739702856,
      -0.12825907397806091,  0.21881316822144872,   -0.026042238957297705, -0.171888066028661,
      0.18951348041519955,
  }};
  const std::array<double, 49> density{{
      3549452.3026884324,  -7743947.1606169157, -13934125.422198512, -5847323.7535195546,
      4070234.7480357229,  944779.87302844413,  -5007269.8183510015, -7743947.1606169157,
      31240812.767640717,  59239016.207355976,  28152897.214950942,  -11690076.105434233,
      -2986181.3240083936, 23306262.439213812,  -13934125.422198512, 59239016.207355976,
      112835115.70717451,  53962409.433979936,  -21565441.380274918, -5551334.3880044529,
      44612781.893574461,  -5847323.7535195546, 28152897.214950942,  53962409.433979936,
      26176367.997345824,  -9698445.6248420794, -2542903.0392612224, 21560498.133406397,
      4070234.7480357229,  -11690076.105434233, -21565441.380274918, -9698445.6248420794,
      5241609.2554276148,  1271091.2550357149,  -8142163.7820452452, 944779.87302844413,
      -2986181.3240083936, -5551334.3880044529, -2542903.0392612224, 1271091.2550357149,
      312900.50221088144,  -2124259.9134732755, -5007269.8183510015, 23306262.439213812,
      44612781.893574461,  21560498.133406397,  -8142163.7820452452, -2124259.9134732755,
      17776919.043332942,
  }};
  Input input{7, 7, 3, 2, metric, metric, metric, density};
  const auto result = project_density(input);
  require(result.diagnostics.source_metric_rank == 7 && result.diagnostics.target_metric_rank == 7,
          "retained metric modes were silently discarded to pass the purity gate");
  for (std::size_t index = 0; index < density.size(); ++index)
    close(result.density[index], density[index], 1e-7 * (1.0 + std::abs(density[index])));
  close(result.diagnostics.electron_trace, 6, 1e-7);
  require(result.diagnostics.metric_orthogonality_error < 1e-7,
          "tight metric eigensolve failed occupied orthogonality");
  auto fractional = density;
  for (auto& value : fractional) value *= 0.99;
  input.source_state = fractional;
  rejected([&] { (void)project_density(input); });
}

void accurate_source_gate_regressions() {
  // All literals are FP64 inputs. A 70-digit independent oracle gives this
  // condition-9e8 density a worst population error 6.7089414221e-8, inside the
  // unchanged 1e-7 gate. Its complete target span has the independent density 2I.
  const std::array<double, 4> source{
      {0.08511214162043723, 0.2790484974922419, 0.2790484974922419, 0.9148878594906736}};
  const std::array<double, 4> state{
      {1646798147.083212, -502287295.4860362, -502287295.4860362, 153201854.91678748}};
  const std::array<double, 4> cross{
      {0.29173985264347624, 0.9564976980819144, 0.0, 0.00011425704601985191}};
  const std::array<double, 4> unit{{1, 0, 0, 1}};
  for (double scale : {1e-200, 1.0, 1e200}) {
    for (double sign : {-1.0, 1.0}) {
      auto metric = source;
      auto density = state;
      auto overlap = cross;
      for (std::size_t row = 0; row < 2; ++row)
        for (std::size_t column = 0; column < 2; ++column) {
          const auto index = row * 2 + column;
          const auto phase = row == column ? 1.0 : sign;
          metric[index] *= scale * phase;
          density[index] *= phase / scale;
          overlap[index] *= std::sqrt(scale) * (column == 1 ? sign : 1.0);
        }
      const auto result = project_density({2, 2, 2, 2, metric, unit, overlap, density});
      for (std::size_t index = 0; index < 4; ++index) close(result.density[index], 2 * unit[index]);
      close(result.diagnostics.electron_trace, 4);
    }
  }

  // Purity passes (4.2616432278e-8), but the independent elementwise
  // reconstruction error is 1.1724260433e-7. Do not merge these two gates.
  const std::array<double, 36> reconstruction_metric{{
      0.091577318178850523,  -0.024244397571812449, 0.19420763635224425,   -0.02876352135550285,
      0.13260381852440217,   0.16257703503546658,   -0.024244397571812449, 0.015568668775605145,
      -0.04835289448512417,  0.0071641143155514678, -0.033447406114853626, -0.050468914907644477,
      0.19420763635224425,   -0.04835289448512417,  0.41292577854966322,   -0.061200120653505802,
      0.28173141444723604,   0.3421888480630802,    -0.02876352135550285,  0.0071641143155514678,
      -0.061200120653505802, 0.009128889146679817,  -0.041680310391521737, -0.05056506777527147,
      0.13260381852440217,   -0.033447406114853626, 0.28173141444723604,   -0.041680310391521737,
      0.19234690465410312,   0.2341593966988651,    0.16257703503546658,   -0.050468914907644477,
      0.3421888480630802,    -0.05056506777527147,  0.2341593966988651,    0.29490519817240857,
  }};
  const std::array<double, 36> reconstruction_density{{
      86481438.21411407,   -107707529.40679698, -101886276.67096081, 1230984.654383807,
      223201590.16808972,  -124907325.02591598, -107707529.40679698, 186700433.42931536,
      170625062.79513192,  11474139.307076136,  -397410743.70898199, 210874997.1387684,
      -101886276.67096081, 170625062.79513192,  156605659.73014909,  9644253.5312309191,
      -362552970.6846205,  193189546.65842512,  1230984.654383807,   11474139.307076136,
      9644253.5312309191,  3656080.3522149604,  -26703736.111999925, 11925021.532069085,
      223201590.16808972,  -397410743.70898199, -362552970.6846205,  -26703736.111999925,
      847707793.03883004,  -448070703.6703279,  -124907325.02591598, 210874997.1387684,
      193189546.65842512,  11925021.532069085,  -448070703.6703279,  238615311.04646423,
  }};
  const std::array<double, 36> reconstruction_cross{{
      0.30261744526522349,
      -0.080115664021166696,
      0.64175955283091668,
      -0.095049118302791794,
      0.43818960406655999,
      0.53723616261772,
      0,
      0.095656411986090512,
      0.032011426816813848,
      -0.0047127934504660081,
      0.017337520273623824,
      -0.07765169996573193,
      0,
      0,
      0.0067619119394481415,
      -0.0074798004386641952,
      -0.0053163477167336741,
      -0.015062265424820151,
      0,
      0,
      0,
      0.0040492492924874834,
      0.0027580560387034225,
      0.0049729315030200526,
      0,
      0,
      0,
      0,
      0.00056169982940239174,
      0.0010528211201033977,
      0,
      0,
      0,
      0,
      0,
      8.3191675266637498e-05,
  }};
  std::array<double, 36> reconstruction_target{};
  for (std::size_t index = 0; index < 6; ++index) reconstruction_target[index * 6 + index] = 1;
  rejected([&] {
    (void)project_density({6, 6, 5, 2, reconstruction_metric, reconstruction_target,
                           reconstruction_cross, reconstruction_density});
  });
  // Reconstruction passes (6.4354354505e-8), but the independent
  // population error is 1.2302646982e-7. Do not widen the public purity gate.
  const std::array<double, 25> purity_metric{{
      0.0050417437343412514, -0.039233238643405645, -0.031925296828462048, 0.040177143419387558,
      -0.028766587400709739, -0.039233238643405645, 0.30653828054748755,   0.24843118293276542,
      -0.31333172481346289,  0.2225035931742394,    -0.031925296828462048, 0.24843118293276542,
      0.20347957800941038,   -0.255179150028922,    0.18449831600114749,   0.040177143419387558,
      -0.31333172481346289,  -0.255179150028922,    0.32099665090083196,   -0.22985463226597802,
      -0.028766587400709739, 0.2225035931742394,    0.18449831600114749,   -0.22985463226597802,
      0.16975077639435907,
  }};
  const std::array<double, 25> purity_density{{
      77007950.354837105, 176762798.76383349, 120186300.41464061, 274525399.31713253,
      22454464.278207082, 176762798.76383349, 433288536.45120537, 306347423.08570451,
      680879979.81279087, 51012667.798631996, 120186300.41464061, 306347423.08570451,
      221434179.00973308, 484658390.5898692,  34407443.9854187,   274525399.31713253,
      680879979.81279087, 484658390.5898692,  1072146103.9914496, 79045343.52142033,
      22454464.278207082, 51012667.798631996, 34407443.9854187,   79045343.52142033,
      6575883.44834946,
  }};
  const std::array<double, 25> purity_cross{{
      0.071005237372613939,
      -0.55254006739702188,
      -0.44961890150338879,
      0.5658335202592184,
      -0.40513331783895629,
      0,
      0.035181734868850491,
      -3.6246276561530721e-05,
      -0.019499703368738498,
      -0.038338006192832549,
      0,
      0,
      0.036365094617005586,
      -0.021185453136635394,
      0.06438396728115546,
      0,
      0,
      0,
      0.00012804616086920285,
      -0.0015391776429731592,
      0,
      0,
      0,
      0,
      0.0005514903798760839,
  }};
  std::array<double, 25> purity_target{};
  for (std::size_t index = 0; index < 5; ++index) purity_target[index * 5 + index] = 1;
  rejected([&] {
    (void)project_density({5, 5, 5, 2, purity_metric, purity_target, purity_cross, purity_density});
  });
}

void refined_basis_boundary_regressions() {
  // Exact FP64 fixtures independently evaluated at 80 digits. The first has
  // reconstruction error 9.9664452073e-8; ordinary eigenvector Gram error
  // tipped it over 1e-7 without accurate orthonormalization before refinement.
  // The second has reconstruction error 9.0593170606e-8 and distinguishes
  // accurate density congruence from an ordinary rounded contraction.
  {
    const std::array<double, 25> source{
        {0.51085407092586566,  -0.16424767874183976, -0.41671224428985881, -0.12161646204490799,
         -0.18564421002665207, -0.16424767874183976, 0.054602914994953206, 0.13266876265134117,
         0.041392036170862019, 0.059307499910264058, -0.41671224428985881, 0.13266876265134117,
         0.34087810968977128,  0.097539418156098567, 0.15170634060821339,  -0.12161646204490799,
         0.041392036170862019, 0.097539418156098567, 0.031915847166977929, 0.043688185138748253,
         -0.18564421002665207, 0.059307499910264058, 0.15170634060821339,  0.043688185138748253,
         0.067556086808862034}};
    const std::array<double, 9> target{
        {0.2518225851750937, 0.15017401949700893, -0.40723961561186212, 0.15017401949700893,
         0.08960061432303458, -0.24289619743277502, -0.40723961561186212, -0.24289619743277502,
         0.65861013494631637}};
    const std::array<double, 15> cross{
        {0.019403607789470579, -0.008813814270031365, -0.013959802299608345, -0.0080328428026780613,
         -0.006500722786855216, 0.011312310230106655, -0.0051080911111931054,
         -0.0081579949620447893, -0.0046325947153295702, -0.0038039147879468261,
         -0.031152595430822768, 0.014123268892366016, 0.022430075901931069, 0.012851379602905067,
         0.010449441027736666}};
    const std::array<double, 25> density{
        {280796487.44186324,  188123551.72261035, 234544787.87012264, 240882.16887010226,
         79607087.671049476,  188123551.72261035, 126687808.79720119, 157327747.94249037,
         -337580.56257370481, 52653456.27560053,  234544787.87012264, 157327747.94249037,
         195968592.62845808,  54775.846559263562, 66292518.729503587, 240882.16887010226,
         -337580.56257370481, 54775.846559263562, 382181.50740035705, 589370.48830488452,
         79607087.671049476,  52653456.27560053,  66292518.729503587, 589370.48830488452,
         23285005.902463589}};
    const std::array<double, 9> expected{
        {718737933.88941407, 584432739.50493777, 659958034.26132643, 584432739.50493777,
         475323119.57285368, 536673099.49902207, 659958034.26132643, 536673099.49902207,
         605998742.12055469}};
    const auto result = project_density({5, 3, 3, 2, source, target, cross, density},
                                        Policy{1e-10, 1e-7, 1.0, 4096});
    for (std::size_t index = 0; index < expected.size(); ++index)
      close(result.density[index], expected[index], 2e-7 * (1.0 + std::abs(expected[index])));
    close(result.diagnostics.minimum_projected_norm_squared, 0.00021741124243419447, 2e-8);
    close(result.diagnostics.projection_residual, 0.99989128846968445, 2e-8);
    close(result.diagnostics.electron_trace, 6, 2e-7);
  }
  {
    const std::array<double, 49> source{
        {0.020913267485749971,   -0.0072120731101826081, -0.017562120571661604,
         -0.0083374090342314966, 0.040119826631074919,   0.023708544730694982,
         -0.0077576238420425121, -0.0072120731101826081, 0.0068292630644027569,
         -0.015509254567730366,  -0.011355261498628198,  0.022851100134405177,
         0.031012349664620027,   -0.013425060017159259,  -0.017562120571661604,
         -0.015509254567730366,  0.12221939151467964,    0.07789293653559981,
         -0.21655597476220217,   -0.21502734664126263,   0.086755426343577552,
         -0.0083374090342314966, -0.011355261498628198,  0.07789293653559981,
         0.05032398240732365,    -0.13636773104967992,   -0.13835753162695888,
         0.055699224015966699,   0.040119826631074919,   0.022851100134405177,
         -0.21655597476220217,   -0.13636773104967992,   0.38837897155409568,
         0.37726699966119598,    -0.15175239116635783,   0.023708544730694982,
         0.031012349664620027,   -0.21502734664126263,   -0.13835753162695888,
         0.37726699966119598,    0.38136024709528799,    -0.15421560150255256,
         -0.0077576238420425121, -0.013425060017159259,  0.086755426343577552,
         0.055699224015966699,   -0.15175239116635783,   -0.15421560150255256,
         0.063228042269198592}};
    const std::array<double, 25> target{
        {0.12478143301320495,  -0.25349603952686062, -0.057381785133186571, -0.15336614636558252,
         -0.13008044802276245, -0.25349603952686062, 0.51914141556501991,   0.1199966625957458,
         0.31749779184237559,  0.26479841025784256,  -0.057381785133186571, 0.1199966625957458,
         0.0292077024983886,   0.07540926860367822,  0.060256630596260997,  -0.15336614636558252,
         0.31749779184237559,  0.07540926860367822,  0.19696036972331418,   0.1606555834420042,
         -0.13008044802276245, 0.26479841025784256,  0.060256630596260997,  0.1606555834420042,
         0.13571610878650325}};
    const std::array<double, 35> cross{
        {-0.013974797529091596,  0.010045734785885314,  -0.01225028995448787,
         -0.0087639642335928438, 0.015413369043190288,  0.027285353645211154,
         -0.014911188329960667,  0.024732792569225261,  -0.019520733064684333,
         0.030476567877581628,   0.020851839902035822,  -0.0427217722698605,
         -0.063821026475022913,  0.033575956444365423,  0.0034308216601192233,
         -0.003883694176815129,  0.010169955532915206,  0.0065033005205490015,
         -0.016357430827452071,  -0.019346984806082483, 0.0095088449783508298,
         0.011962427678941187,   -0.011124994069519875, 0.023242070553001284,
         0.015249168630894901,   -0.035587460636010211, -0.04587900677395465,
         0.023183345086387294,   0.014089423624834634,  -0.010528791580099032,
         0.014372196617304512,   0.010066079699311125,  -0.019083235344412586,
         -0.03107440544765571,   0.016681160678021348}};
    const std::array<double, 49> density{
        {6684614.2829120168,  -466002.58361657098, 41563935.849233873,  -35694411.935743764,
         5190735.8448847001,  -3215469.9372610473, -20244257.295005523, -466002.58361657098,
         8649922.4819736797,  2977554.5490556327,  -776002.23364830844, 5496538.7705057766,
         -2638443.3165794667, 5136127.2632589675,  41563935.849233873,  2977554.5490556327,
         264596010.33005992,  -225559719.99958572, 36890199.984195217,  -22032211.12725687,
         -123786821.40835162, -35694411.935743764, -776002.23364830844, -225559719.99958572,
         192778289.34559271,  -30340517.937968303, 18334767.716370344,  106997087.54159726,
         5190735.8448847001,  5496538.7705057766,  36890199.984195217,  -30340517.937968303,
         8192701.7343595652,  -4469532.3599570105, -13318751.654372714, -3215469.9372610473,
         -2638443.3165794667, -22032211.12725687,  18334767.716370344,  -4469532.3599570105,
         2515164.4465202624,  8528677.8422734048,  -20244257.295005523, 5136127.2632589675,
         -123786821.40835162, 106997087.54159726,  -13318751.654372714, 8528677.8422734048,
         63020595.783374391}};
    const std::array<double, 25> expected{
        {452058363.78554577, 339190266.13609833, -30231503.76031372,  -224988620.14571738,
         51252907.320733331, 339190266.13609833, 254514716.04911962,  -22623974.358482771,
         -168857537.8585287, 38457329.378452726, -30231503.76031372,  -22623974.358482771,
         5620644.3703470975, 12848611.66060661,  -2539319.9080191837, -224988620.14571738,
         -168857537.8585287, 12848611.66060661,  113322610.05768771,  -26042551.168303389,
         51252907.320733331, 38457329.378452726, -2539319.9080191837, -26042551.168303389,
         6046989.7276845146}};
    const auto result = project_density({7, 5, 4, 1, source, target, cross, density},
                                        Policy{1e-10, 1e-7, 1.0, 4096});
    for (std::size_t index = 0; index < expected.size(); ++index)
      close(result.density[index], expected[index], 2e-7 * (1.0 + std::abs(expected[index])));
    close(result.diagnostics.minimum_projected_norm_squared, 0.21491218306334395, 2e-8);
    close(result.diagnostics.projection_residual, 0.88605181391194954, 2e-8);
    close(result.diagnostics.electron_trace, 4, 2e-7);
  }
}

void run() {
  refined_basis_boundary_regressions();
  accurate_source_gate_regressions();
  retained_metric_purity_regression();
  // Tightening numerical execution must not relax public purity acceptance.
  const std::array<double, 1> unit{{1}};
  const std::array<double, 1> within_purity_tolerance{{2.0 + 5e-8}};
  const std::array<double, 1> outside_purity_tolerance{{2.0 + 2e-7}};
  Input boundary{1, 1, 1, 2, unit, unit, unit, within_purity_tolerance};
  close(project_density(boundary).diagnostics.electron_trace, 2);
  boundary.source_state = outside_purity_tolerance;
  rejected([&] { (void)project_density(boundary); });
  const std::array<double, 4> source{{1, 0, 0, 1}};
  const std::array<double, 9> target{{1, 0, 0, 0, 1, 0, 0, 0, 1}};
  const std::array<double, 6> cross{{0.8, 0, 0, 0.6, 0, 0.8}};
  const std::array<double, 9> expected{{2, 0, 0, 0, 0.72, 0.96, 0, 0.96, 1.28}};
  const Policy permissive{1e-10, 1e-7, 1.0, 4096};
  Input input{2, 3, 2, 2, source, target, cross, source};
  auto result = project_occupied(input, permissive);
  require(result.density.size() == expected.size() && result.coefficients.size() == 6,
          "detached occupied projection shapes differ");
  for (std::size_t index = 0; index < expected.size(); ++index)
    close(result.density[index], expected[index]);
  close(result.diagnostics.minimum_projected_norm_squared, 0.64);
  close(result.diagnostics.projection_residual, 0.6);
  close(result.diagnostics.electron_trace, 4);
  close(result.diagnostics.metric_idempotency_error, 0);
  require(result.diagnostics.source_metric_rank == 2 && result.diagnostics.target_metric_rank == 3,
          "metric ranks were not retained");
  const auto detached = result;
  const std::array<double, 4> rotated{{0.6, -0.8, 0.8, 0.6}};
  input.source_state = rotated;
  result = project_occupied(input, permissive);
  for (std::size_t index = 0; index < expected.size(); ++index)
    close(result.density[index], detached.density[index]);
  input.source_state = source;
  rejected([&] { (void)project_occupied(input); });

  const std::array<double, 4> density{{2, 0, 0, 2}};
  input.source_state = density;
  result = project_density(input, permissive);
  for (std::size_t index = 0; index < expected.size(); ++index)
    close(result.density[index], expected[index]);
  const std::array<double, 4> fractional{{2, 0, 0, 1.8}};
  input.source_state = fractional;
  rejected([&] { (void)project_density(input, permissive); });

  input.occupied = 0;
  input.occupation = 1;
  const std::array<double, 4> zero{};
  input.source_state = zero;
  result = project_density(input, permissive);
  require(result.coefficients.empty(), "empty spin channel invented orbitals");
  for (const auto value : result.density) close(value, 0);
  close(result.diagnostics.electron_trace, 0);
  close(result.diagnostics.minimum_projected_norm_squared, 1);
  close(result.diagnostics.projection_residual, 0);
  auto all_discarded = permissive;
  all_discarded.relative_threshold = 1;
  result = project_density(input, all_discarded);
  require(result.diagnostics.source_metric_rank == 0 && result.diagnostics.target_metric_rank == 0,
          "inclusive metric cutoff retained a boundary mode");

  const std::array<double, 4> ill_conditioned{{1, 0, 0, 1e-10}};
  input = {2, 2, 2, 2, ill_conditioned, source, source, source};
  rejected([&] { (void)project_occupied(input, permissive); });
  const std::array<double, 4> asymmetric{{1, 0.01, 0, 1}};
  input = {2, 2, 2, 2, asymmetric, source, source, source};
  rejected([&] { (void)project_occupied(input, permissive); });
  const std::array<double, 4> indefinite{{1, 0, 0, -1}};
  input.source_overlap = indefinite;
  rejected([&] { (void)project_occupied(input, permissive); });
  input = {2, 2, 2, 2, source, source, source, source};
  const std::array<double, 4> inconsistent{{1.1, 0, 0, 1}};
  input.cross_overlap = inconsistent;
  rejected([&] { (void)project_occupied(input, permissive); });
  const std::array<double, 4> nonfinite{{1, 0, 0, std::numeric_limits<double>::quiet_NaN()}};
  input.cross_overlap = nonfinite;
  rejected([&] { (void)project_occupied(input, permissive); });
  input.cross_overlap = std::span<const double>(source).first(3);
  rejected([&] { (void)project_occupied(input, permissive); });

  const std::array<double, 4> nontrivial_metric{{4, 0, 0, 9}};
  const std::array<double, 4> normalized{{0.5, 0, 0, 1.0 / 3.0}};
  input = {2, 2, 2, 2, nontrivial_metric, nontrivial_metric, nontrivial_metric, normalized};
  result = project_occupied(input);
  close(result.density[0], 0.5);
  close(result.density[3], 2.0 / 9.0);
  close(result.diagnostics.electron_trace, 4);
  close(result.diagnostics.normal_equation_residual, 0);

  const std::array<double, 4> half_cross{{0.5, 0, 0, 0.5}};
  input = {2, 2, 2, 2, source, source, half_cross, source};
  auto inclusive = permissive;
  inclusive.relative_threshold = 0.25;
  rejected([&] { (void)project_occupied(input, inclusive); });
  inclusive.relative_threshold = std::nextafter(0.25, 0.0);
  result = project_occupied(input, inclusive);
  close(result.diagnostics.minimum_projected_norm_squared, 0.25);
  inclusive.maximum_residual = std::sqrt(0.75);
  result = project_occupied(input, inclusive);
  close(result.diagnostics.projection_residual, inclusive.maximum_residual);
  inclusive.maximum_residual = std::nextafter(inclusive.maximum_residual, 0.0);
  rejected([&] { (void)project_occupied(input, inclusive); });

  for (const double scale : {1e-200, 1e200}) {
    const std::array<double, 4> scaled_metric{{scale, 0, 0, scale}};
    const double coefficient = 1.0 / std::sqrt(scale);
    const std::array<double, 4> scaled_coefficients{{coefficient, 0, 0, coefficient}};
    input = {2, 2, 2, 2, scaled_metric, scaled_metric, scaled_metric, scaled_coefficients};
    result = project_occupied(input);
    close(result.density[0] * scale, 2);
    close(result.density[3] * scale, 2);
    close(result.diagnostics.electron_trace, 4);
    close(result.diagnostics.minimum_projected_norm_squared, 1);
  }
}
}  // namespace

int main() {
  try {
    run();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
