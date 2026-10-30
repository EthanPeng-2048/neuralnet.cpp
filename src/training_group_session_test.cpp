// ── training_group_session_test — 通用多模型 role 生命周期（RL-2a）────────
//
// 验收：
//   [1] 两个可训练 role + 一个只读 role，统一引擎、可查询；
//   [2] 不同角色的 output schema 合法共存；
//   [3] 重复/空 role、跨引擎、未知 role、只读 role 误取 Session 均报错；
//   [4] group move 后 Session/optimizer TensorRef 仍可训练。
// ─────────────────────────────────────────────────────────────────────────

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "neuralnet.cpp/nn.hpp"
#include "test_common.hpp"

int g_fail = 0;

namespace
{

nn::TensorSchema role_schema(std::string name)
{
    auto r = nn::TensorSchema::create(std::move(name), {nn::tensor_field("output")});
    NN_ASSERT_ALWAYS(r, "role_schema");
    return std::move(*r);
}

nn::Result<nn::Model> make_model(nn::CpuEngine& engine, std::string role, std::size_t in = 3,
                                 std::size_t out = 2)
{
    nn::Model model(engine);
    model.set_output_schema(role_schema("role_" + role));
    NN_TRY_CHECK(model.add<nn::Linear>(in, out));
    return model;
}

nn::Result<nn::TrainingSession> make_session(nn::CpuEngine& engine, std::string role)
{
    NN_TRY(model, make_model(engine, std::move(role)));
    nn::TrainingConfig cfg;
    cfg.optimizer = "adamw";
    cfg.lr = 0.01f;
    return nn::TrainingSession::create(std::move(*model), cfg);
}

void train_one_microstep(nn::TrainingSession& session)
{
    const auto x = make_tensor(session.engine(), 3, 2, 0.2f, 0.1f);
    nn::TrainingStepOptions opts;
    opts.normalization_units = 2;
    auto report =
        session.train_step(x, opts,
                           [](nn::Model& model, nn::ComputeEngine& engine,
                              const nn::Tensor& input) -> nn::Result<nn::TrainingStepOutput>
                           {
                               NN_TRY(out, model.forward(input));
                               nn::MSELoss loss;
                               auto target =
                                   make_tensor(engine, out->rows(), out->cols(), 0.0f, 0.0f);
                               NN_TRY_CHECK(loss.forward(engine, *out, target));
                               NN_TRY(grad, loss.backward());
                               return nn::TrainingStepOutput{
                                   .grad = std::move(*grad),
                                   .gradient_units = 2,
                                   .loss_sum = nn::Tensor{},
                                   .loss_units = 0,
                               };
                           });
    CHECK(report && report->parameters_updated, "group role executes one optimizer update");
}

void test_roles_and_lifecycle()
{
    auto engine = nn::CpuEngine::create();
    auto actor = make_session(*engine, "actor");
    auto critic = make_session(*engine, "critic");
    auto reference = make_model(*engine, "reference");
    CHECK(actor && critic && reference, "build three role models");
    if (!actor || !critic || !reference) return;

    nn::TrainingGroupSession group;
    CHECK(group.add_training_role("actor", std::move(*actor)).has_value(),
          "add actor training role");
    CHECK(group.add_training_role("critic", std::move(*critic)).has_value(),
          "add critic training role");
    CHECK(group.add_readonly_role("reference", std::move(*reference)).has_value(),
          "add readonly reference role");
    CHECK(group.size() == 3, "group role count");
    CHECK((group.role_names() == std::vector<std::string>{"actor", "critic", "reference"}),
          "group preserves add order");
    CHECK(group.engine_bound() && &group.engine() == engine.get(), "group reports shared engine");

    auto actor_trainable = group.is_trainable("actor");
    auto ref_trainable = group.is_trainable("reference");
    CHECK(actor_trainable && *actor_trainable, "actor is trainable");
    CHECK(ref_trainable && !*ref_trainable, "reference is readonly");
    CHECK(group.training_session("actor").has_value(), "training role resolves Session");
    CHECK(!group.training_session("reference").has_value(), "readonly role cannot resolve Session");
    CHECK(group.model("reference").has_value(), "readonly role resolves Model");
    CHECK(!group.model("missing").has_value(), "unknown role rejected");

    auto actor_schema = group.schema("actor");
    auto critic_schema = group.schema("critic");
    CHECK(actor_schema && critic_schema, "role schemas query");
    if (actor_schema && critic_schema)
        CHECK(actor_schema->fingerprint() != critic_schema->fingerprint(),
              "different role schemas may coexist");

    nn::TrainingGroupSession moved = std::move(group);
    CHECK(moved.size() == 3 && group.empty(), "group move transfers role ownership");
    auto moved_actor = moved.training_session("actor");
    CHECK(moved_actor.has_value(), "moved group resolves training role");
    if (moved_actor) train_one_microstep(**moved_actor);
}

void test_rejections()
{
    auto engine = nn::CpuEngine::create();
    auto actor = make_session(*engine, "actor");
    CHECK(actor.has_value(), "rejection actor");
    if (!actor) return;
    nn::TrainingGroupSession group;
    CHECK(group.add_training_role("actor", std::move(*actor)).has_value(), "rejection actor added");
    auto duplicate = make_session(*engine, "actor");
    CHECK(duplicate.has_value(), "duplicate source session");
    if (duplicate)
        CHECK(!group.add_training_role("actor", std::move(*duplicate)), "duplicate role rejected");

    auto blank = make_session(*engine, "blank");
    CHECK(blank.has_value(), "blank source session");
    if (blank) CHECK(!group.add_training_role("", std::move(*blank)), "empty role rejected");

    auto other_engine = nn::CpuEngine::create();
    auto other = make_session(*other_engine, "other");
    CHECK(other.has_value(), "other-engine session");
    if (other)
        CHECK(!group.add_training_role("other", std::move(*other)), "cross-engine role rejected");

    auto readonly = make_model(*engine, "help");
    CHECK(readonly.has_value(), "readonly model");
    if (readonly)
    {
        CHECK(group.add_readonly_role("help", std::move(*readonly)).has_value(),
              "readonly role added");
        CHECK(!group.add_readonly_role("help", NN_UNWRAP(make_model(*engine, "help2"))),
              "duplicate readonly role rejected");
    }
}

} // namespace

int main()
{
    std::puts("=== RL-2a TrainingGroupSession ===");
    test_roles_and_lifecycle();
    test_rejections();
    if (g_fail == 0) std::puts("All TrainingGroupSession tests passed.");
    else std::printf("%d TrainingGroupSession test(s) failed.\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
