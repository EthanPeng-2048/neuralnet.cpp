#pragma once

// ── training_group_session.hpp — 通用多模型生命周期容器（RL-2a）──────────
//
// 只回答“哪些具名角色存在、谁拥有它们、是否可训练”。不解释 actor/critic，
// 不拥有 rollout/buffer/RNG/checkpoint 策略；算法行为留在编排层。
//
// 生命周期：
//   · 每个 TrainingSession 内部仍保证 optimizer 先于 Model 析构；
//   · role 节点堆分配 → group move 不改 Session/Model 地址；
//   · group 内所有角色共用同一 ComputeEngine（跨引擎组合初始化期拒绝）。
// ─────────────────────────────────────────────────────────────────────────

#include <algorithm>
#include <cstddef>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "compute_engine.hpp"
#include "model_container.hpp"
#include "training_session.hpp"

namespace nn
{

class TrainingGroupSession
{
private:
    struct Role
    {
        std::string name;
        std::unique_ptr<TrainingSession> training;
        std::unique_ptr<Model> readonly;

        [[nodiscard]] bool trainable() const noexcept { return training != nullptr; }

        [[nodiscard]] Model& model() { return training ? training->model() : *readonly; }

        [[nodiscard]] const Model& model() const
        { return training ? training->model() : *readonly; }
    };

    ComputeEngine* engine_ = nullptr;
    std::vector<std::unique_ptr<Role>> roles_;

    [[nodiscard]] Role* find_role(std::string_view name) noexcept
    {
        for (auto& role : roles_)
            if (role->name == name) return role.get();
        return nullptr;
    }

    [[nodiscard]] const Role* find_role(std::string_view name) const noexcept
    {
        for (const auto& role : roles_)
            if (role->name == name) return role.get();
        return nullptr;
    }

    [[nodiscard]] Result<void> validate_new_role(std::string_view name, Model& model)
    {
        if (name.empty()) NN_FAIL("TrainingGroupSession: role 名不能为空");
        if (find_role(name) != nullptr)
            NN_FAIL("TrainingGroupSession: role '" + std::string(name) + "' 重复");
        if (!model.engine_bound())
            NN_FAIL("TrainingGroupSession: role '" + std::string(name) + "' 的 Model 未绑定引擎");
        ComputeEngine& candidate = model.engine();
        if (engine_ == nullptr) engine_ = &candidate;
        else if (engine_ != &candidate)
            NN_FAIL("TrainingGroupSession: 所有 role 必须共享同一 ComputeEngine"
                    "（role='" +
                    std::string(name) + "'）");

        // 有效性校验；不同算法角色可拥有不同 schema，不强制 fingerprint 相同。
        NN_TRY(schema, model.output_schema());
        if (!schema->valid())
            NN_FAIL("TrainingGroupSession: role '" + std::string(name) + "' 的 output schema 无效");
        return {};
    }

public:
    TrainingGroupSession() = default;
    TrainingGroupSession(const TrainingGroupSession&) = delete;
    TrainingGroupSession& operator=(const TrainingGroupSession&) = delete;
    TrainingGroupSession(TrainingGroupSession&&) noexcept = default;
    TrainingGroupSession& operator=(TrainingGroupSession&&) noexcept = default;
    ~TrainingGroupSession() = default;

    [[nodiscard]] Result<void> add_training_role(std::string name, TrainingSession session)
    {
        NN_TRY_CHECK(validate_new_role(name, session.model()));
        auto role = std::make_unique<Role>();
        role->name = std::move(name);
        role->training = std::make_unique<TrainingSession>(std::move(session));
        roles_.push_back(std::move(role));
        return {};
    }

    [[nodiscard]] Result<void> add_readonly_role(std::string name, Model model)
    {
        NN_TRY_CHECK(validate_new_role(name, model));
        auto role = std::make_unique<Role>();
        role->name = std::move(name);
        role->readonly = std::make_unique<Model>(std::move(model));
        roles_.push_back(std::move(role));
        return {};
    }

    [[nodiscard]] bool empty() const noexcept { return roles_.empty(); }
    [[nodiscard]] std::size_t size() const noexcept { return roles_.size(); }
    [[nodiscard]] bool engine_bound() const noexcept { return engine_ != nullptr; }
    [[nodiscard]] ComputeEngine& engine() const
    {
        NN_ASSERT_ALWAYS(engine_ != nullptr, "TrainingGroupSession::engine: group 为空");
        return *engine_;
    }

    [[nodiscard]] std::vector<std::string> role_names() const
    {
        std::vector<std::string> names;
        names.reserve(roles_.size());
        for (const auto& role : roles_)
            names.push_back(role->name);
        return names;
    }

    [[nodiscard]] Result<bool> is_trainable(std::string_view name) const
    {
        const Role* role = find_role(name);
        if (role == nullptr) NN_FAIL("TrainingGroupSession: 未知 role '" + std::string(name) + "'");
        return role->trainable();
    }

    [[nodiscard]] Result<Model*> model(std::string_view name)
    {
        Role* role = find_role(name);
        if (role == nullptr) NN_FAIL("TrainingGroupSession: 未知 role '" + std::string(name) + "'");
        return &role->model();
    }

    [[nodiscard]] Result<const Model*> model(std::string_view name) const
    {
        const Role* role = find_role(name);
        if (role == nullptr) NN_FAIL("TrainingGroupSession: 未知 role '" + std::string(name) + "'");
        return &role->model();
    }

    [[nodiscard]] Result<TrainingSession*> training_session(std::string_view name)
    {
        Role* role = find_role(name);
        if (role == nullptr) NN_FAIL("TrainingGroupSession: 未知 role '" + std::string(name) + "'");
        if (role->training == nullptr)
            NN_FAIL("TrainingGroupSession: role '" + std::string(name) +
                    "' 是只读 Model，不是 TrainingSession");
        return role->training.get();
    }

    [[nodiscard]] Result<const TrainingSession*> training_session(std::string_view name) const
    {
        const Role* role = find_role(name);
        if (role == nullptr) NN_FAIL("TrainingGroupSession: 未知 role '" + std::string(name) + "'");
        if (role->training == nullptr)
            NN_FAIL("TrainingGroupSession: role '" + std::string(name) +
                    "' 是只读 Model，不是 TrainingSession");
        return role->training.get();
    }

    [[nodiscard]] Result<TensorSchema> schema(std::string_view name) const
    {
        NN_TRY(role, model(name));
        return (*role)->output_schema();
    }
};

} // namespace nn
