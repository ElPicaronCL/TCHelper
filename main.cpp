// Gta Sa Dynamic Override - full implementation with OpenBVE train sounds by vehicle model name
// Uses Plugin-SDK (DK22Pac) conventions where possible.
// Copyright Digital XP Studios | Sebastian Patricio Tapia Moya AKA: ElPicaronCL, All rights reserved, 2025
//Powered by GPT-5, Copyright OpenAI 2025

#include "plugin.h"
#include "game_sa/CModelInfo.h"
#include "game_sa/CVehicleModelInfo.h"
#include "game_sa/CPedModelInfo.h"
#include "game_sa/CStreaming.h"
#include "game_sa/CCarCtrl.h"
#include "game_sa/CPopulation.h"
#include "game_sa/CFileMgr.h"
#include "game_sa/CFileLoader.h"
#include "game_sa/CHandlingDataMgr.h"
#include "game_sa/CarGroup.h"
#include "game_sa/CPedType.h"
#include "game_sa/common.h"
#include "game_sa/CTimer.h"
#include "game_sa/CTrain.h"
#include "game_sa/CRenderer.h"
#include "game_sa/CVehicle.h"
#include "game_sa/CPed.h"
#include "game_sa/cDMAudio.h"

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <thread>
#include <mutex>
#include <atomic>
#include <algorithm>
#include <cctype>

using namespace plugin;

// --------------------
// Configuration file paths
// --------------------
static const char* VEHICLES_INI = "MyVehicles.ini";
static const char* PEDS_INI     = "MyPeds.ini";
static const char* CARGRP_INI   = "MyCarGroups.ini";
static const char* PEDGRP_INI   = "MyPedGroups.ini";
static const char* SOUNDS_INI   = "MySounds.ini";
static const char* CONTROLS_INI = "Controls.ini";

// --------------------
// Defaults to generate when INIs missing
// --------------------
static const char* DEF_VEHICLES =
"; MyVehicles.ini sample format\n"
"; Each non-comment line: modelName, handlingId, carGroup, flags\n"
"infernus, HANDLING_SUPER, sport, 0x0\n"
"tahoma, HANDLING_SAL, sedan, 0x0\n";

static const char* DEF_PEDS =
"; MyPeds.ini sample format\n"
"player, PLAYER, civilian, 0x0\n"
"gangb, GANG, gangB, 0x0\n";

static const char* DEF_CARGRP =
"; MyCarGroups.ini sample\n"
"sports: infernus, bullet, cheetah\n";

static const char* DEF_PEDGRP =
"; MyPedGroups.ini sample\n"
"gangs: gangb, gangc\n";

static const char* DEF_SOUNDS =
"; MySounds.ini sample\n"
"default: gta_sa/audio/openbve/train/default/sounds/\n";

static const char* DEF_CONTROLS =
"[KEYS]\n"
"DoorLeftOpen=I\n"
"DoorRightOpen=O\n";

// --------------------
// Simple INI parser (robust enough for our needs)
// --------------------
struct SimpleIni {
    std::map<std::string, std::map<std::string, std::string>> data;
    bool load(const char* path) {
        std::ifstream f(path);
        if (!f.is_open()) return false;
        std::string line, section;
        auto trim = [](std::string &s){
            while (!s.empty() && isspace((unsigned char)s.front())) s.erase(s.begin());
            while (!s.empty() && isspace((unsigned char)s.back())) s.pop_back();
        };
        while (std::getline(f, line)) {
            trim(line);
            if (line.empty() || line[0] == ';' || line[0] == '#') continue;
            if (line.front() == '[' && line.back() == ']') { section = line.substr(1, line.size()-2); continue; }
            auto eq = line.find('=');
            if (eq == std::string::npos) {
                // support key: value lines too (for groups)
                auto colon = line.find(':');
                if (colon != std::string::npos) {
                    std::string key = line.substr(0, colon);
                    std::string val = line.substr(colon+1);
                    trim(key); trim(val);
                    data[section][key] = val;
                }
                continue;
            }
            std::string key = line.substr(0, eq);
            std::string val = line.substr(eq+1);
            trim(key); trim(val);
            data[section][key] = val;
        }
        return true;
    }
    std::string get(const char* section, const char* key, const char* d="") const {
        auto s = data.find(section);
        if (s == data.end()) return d;
        auto k = s->second.find(key);
        if (k == s->second.end()) return d;
        return k->second;
    }
};

// --------------------
// Data models
// --------------------
struct VehicleDef { std::string modelName; std::string handlingId; std::string carGroup; uint32_t flags; };
struct PedDef     { std::string modelName; std::string pedType; std::string voice; uint32_t flags; };
struct TrainSoundSet { std::string basePath; std::map<std::string, std::string> namedSounds; };

// Runtime containers
static std::vector<VehicleDef> g_vehicles;
static std::vector<PedDef> g_peds;
static std::map<std::string, TrainSoundSet> g_trainSounds;

static std::mutex g_mutex;
static std::atomic<bool> g_threadRun{false};
static int g_keyLeft = 'I';
static int g_keyRight = 'O';

// --------------------
// FLA compatibility stubs
// --------------------
extern "C" __declspec(dllexport) void __cdecl modelSpecialFeatures_Stub(int modelIndex) {
    (void)modelIndex; // stub: let FLA handle special features
}
extern "C" __declspec(dllexport) void __cdecl vehicleAudioLoader_Stub(const char* name) {
    (void)name; // stub: FLA handles vehicle audio loading
}

// --------------------
// Helpers
// --------------------

static void EnsureIniFile(const char* path, const char* content) {
    std::ifstream f(path);
    if (!f.good()) {
        std::ofstream out(path);
        out << content;
        out.close();
    }
}

static std::string Trim(const std::string &s) {
    size_t start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    size_t end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

static int FindFreeModelIndex() {
    const int MAX_MODELS = 4000;
    for (int i = 0; i < MAX_MODELS; ++i) {
        if (CModelInfo::GetModelInfo(i) == nullptr) return i;
    }
    return -1;
}

static bool LoadModelFilesByName(const std::string &modelName) {
    int modelId = CModelInfo::GetModelIndex(modelName.c_str());
    if (modelId >= 0) return true; // already loaded
    try {
        char dff[64], txd[64];
        sprintf(dff, "%s.dff", modelName.c_str());
        sprintf(txd, "%s.txd", modelName.c_str());
        CFileLoader::LoadClumpModel(dff, modelName.c_str());
        CFileLoader::LoadTexDictionary(txd);
        return true;
    } catch (...) {
        return false;
    }
}

static int RegisterVehicleModel(const VehicleDef &v) {
    int idx = FindFreeModelIndex();
    if (idx < 0) return -1;

    LoadModelFilesByName(v.modelName);

    CVehicleModelInfo *mi = nullptr;
    try { mi = new CVehicleModelInfo(); } catch (...) { mi = nullptr; }

    if (!mi) {
        CModelInfo *gmi = new CModelInfo();
        gmi->SetModelName(v.modelName.c_str());
        CModelInfo::AddModel(idx, gmi);
        return idx;
    }

    mi->SetModelName(v.modelName.c_str());

    if (!v.handlingId.empty()) {
        int h = CHandlingDataMgr::FindHandlingId(v.handlingId.c_str());
        if (h >= 0) mi->m_nHandlingId = h;
        else {
            CHandlingDataMgr::AddHandlingLine(v.handlingId.c_str());
            mi->m_nHandlingId = CHandlingDataMgr::FindHandlingId(v.handlingId.c_str());
        }
    }

    CModelInfo::AddModel(idx, mi);

    modelSpecialFeatures_Stub(idx);

    return idx;
}

static int RegisterPedModel(const PedDef &p) {
    int idx = FindFreeModelIndex();
    if (idx < 0) return -1;

    LoadModelFilesByName(p.modelName);

    CPedModelInfo *mi = nullptr;
    try { mi = new CPedModelInfo(); } catch (...) { mi = nullptr; }

    if (!mi) {
        CModelInfo *gmi = new CModelInfo();
        gmi->SetModelName(p.modelName.c_str());
        CModelInfo::AddModel(idx, gmi);
        return idx;
    }

    mi->SetModelName(p.modelName.c_str());

    if (!p.pedType.empty()) {
        mi->m_pedType = CPedType::GetPedTypeFromName(p.pedType.c_str());
    }

    CModelInfo::AddModel(idx, mi);

    return idx;
}

static void ClearAllCustomModels() {
    for (int i = 0; i < 4000; ++i) {
        CModelInfo *mi = CModelInfo::GetModelInfo(i);
        if (!mi) continue;
        const char* name = mi->GetModelName();
        if (!name) continue;
        std::string sname(name);
        if (sname.rfind("my_", 0) == 0 || sname.find("infernus") != std::string::npos) {
            CStreaming::RemoveModel(i);
            CModelInfo::RemoveModel(i);
        }
    }
}

// --------------------
// INI loading
// --------------------

static void LoadVehiclesINI() {
    SimpleIni ini;
    if (!ini.load(VEHICLES_INI)) { EnsureIniFile(VEHICLES_INI, DEF_VEHICLES); ini.load(VEHICLES_INI); }
    g_vehicles.clear();

    std::ifstream f(VEHICLES_INI);
    std::string line;
    while (std::getline(f, line)) {
        line = Trim(line);
        if (line.empty() || line[0] == ';' || line[0] == '#') continue;
        std::vector<std::string> parts;
        std::stringstream ss(line);
        std::string tok;
        while (std::getline(ss, tok, ',')) parts.push_back(Trim(tok));
        if (parts.empty()) continue;
        VehicleDef v; v.modelName = parts[0]; if (parts.size()>1) v.handlingId = parts[1]; if (parts.size()>2) v.carGroup = parts[2]; v.flags = 0; g_vehicles.push_back(v);
    }
}

static void LoadPedsINI() {
    std::ifstream f(PEDS_INI);
    if (!f.good()) { EnsureIniFile(PEDS_INI, DEF_PEDS); }
    SimpleIni ini; ini.load(PEDS_INI);
    g_peds.clear();
    std::ifstream ff(PEDS_INI);
    std::string line;
    while (std::getline(ff, line)) {
        line = Trim(line); if (line.empty() || line[0] == ';' || line[0] == '#') continue;
        std::vector<std::string> parts; std::stringstream ss(line); std::string tok; while (std::getline(ss, tok, ',')) parts.push_back(Trim(tok));
        PedDef p; p.modelName = parts.size() > 0 ? parts[0] : ""; p.pedType = parts.size() > 1 ? parts[1] : ""; p.voice = parts.size() > 2 ? parts[2] : ""; p.flags = 0; g_peds.push_back(p);
    }
}

static void LoadSoundsINI() {
    SimpleIni ini;
    if (!ini.load(SOUNDS_INI)) { EnsureIniFile(SOUNDS_INI, DEF_SOUNDS); ini.load(SOUNDS_INI); }
    g_trainSounds.clear();
    for (auto &sec : ini.data) {
        for (auto &kv : sec.second) {
            TrainSoundSet s; s.basePath = kv.second;
            std::ifstream sc(s.basePath + "sound.cfg");
            std::string line;
            while (std::getline(sc, line)) {
                auto eq = line.find('=');
                if (eq == std::string::npos) continue;
                std::string k = Trim(line.substr(0, eq));
                std::string v = Trim(line.substr(eq+1));
                s.namedSounds[k] = s.basePath + v;
            }
            g_trainSounds[kv.first] = s;
        }
    }
}

static void RegisterCarGroups() {
    std::ifstream f(CARGRP_INI);
    if (!f.good()) { EnsureIniFile(CARGRP_INI, DEF_CARGRP); }
    std::string line;
    while (std::getline(f, line)) {
        line = Trim(line); if (line.empty() || line[0] == ';') continue;
        auto colon = line.find(':'); if (colon == std::string::npos) continue;
        std::string group = Trim(line.substr(0, colon)); std::string list = Trim(line.substr(colon+1));
        std::vector<std::string> models; std::stringstream ss(list); std::string tok; while (std::getline(ss, tok, ',')) models.push_back(Trim(tok));
        CarGroup *cg = new CarGroup();
        for (auto &mn : models) {
            int id = CModelInfo::GetModelIndex(mn.c_str());
            if (id < 0) {
                auto it = std::find_if(g_vehicles.begin(), g_vehicles.end(), [&](const VehicleDef &vd){ return _stricmp(vd.modelName.c_str(), mn.c_str())==0; });
                if (it != g_vehicles.end()) RegisterVehicleModel(*it);
                id = CModelInfo::GetModelIndex(mn.c_str());
            }
            if (id >= 0) cg->AddModel(id);
        }
        CPopulation::AddCarGroup(group.c_str(), cg);
    }
}

static void RegisterPedGroups() {
    std::ifstream f(PEDGRP_INI);
    if (!f.good()) { EnsureIniFile(PEDGRP_INI, DEF_PEDGRP); }
    std::string line;
    while (std::getline(f, line)) {
        line = Trim(line); if (line.empty() || line[0] == ';') continue;
        auto colon = line.find(':'); if (colon == std::string::npos) continue;
        std::string group = Trim(line.substr(0, colon)); std::string list = Trim(line.substr(colon+1));
        std::vector<std::string> peds; std::stringstream ss(list); std::string tok; while (std::getline(ss, tok, ',')) peds.push_back(Trim(tok));
        PedGroup *pg = new PedGroup();
        for (auto &pn : peds) {
            int id = CModelInfo::GetModelIndex(pn.c_str());
            if (id < 0) {
                auto it = std::find_if(g_peds.begin(), g_peds.end(), [&](const PedDef &pd){ return _stricmp(pd.modelName.c_str(), pn.c_str())==0; });
                if (it != g_peds.end()) RegisterPedModel(*it);
                id = CModelInfo::GetModelIndex(pn.c_str());
            }
            if (id >= 0) pg->AddModel(id);
        }
        CPopulation::AddPedGroup(group.c_str(), pg);
    }
}

// --------------------
// Sound Handling
// --------------------

struct TrainSoundState {
    bool engineOn = false;
    bool braking = false;
    bool accelerating = false;
    bool doorsOpen = false;
};

static std::map<int, int> g_registeredSamples; // soundName hash -> sampleId
static std::map<int, TrainSoundState> g_trainSoundStates;

// Registers a WAV file as sample if not already registered
static void RegisterWavAsSample(const std::string& soundName, const std::string& path) {
    int hash = std::hash<std::string>{}(soundName);
    if (g_registeredSamples.count(hash) == 0) {
        int sampleId = cDMAudio::LoadSample(path.c_str());
        g_registeredSamples[hash] = sampleId;
    }
}

static std::string GetVehicleModelName(CVehicle* veh) {
    if (!veh) return "";
    int modelIndex = veh->m_nModelIndex;
    CModelInfo* mi = CModelInfo::GetModelInfo(modelIndex);
    if (!mi) return "";
    const char* name = mi->GetModelName();
    return name ? std::string(name) : "";
}

static void PlayTrainSound(CVehicle* veh, const std::string& soundName) {
    if (!veh) return;
    std::string modelName = GetVehicleModelName(veh);
    std::string soundBank = "default";

    if (!modelName.empty()) {
        if (g_trainSounds.find(modelName) != g_trainSounds.end()) {
            soundBank = modelName;
        }
    }

    if (g_trainSounds.find(soundBank) == g_trainSounds.end()) return;
    const TrainSoundSet& set = g_trainSounds[soundBank];
    auto sit = set.namedSounds.find(soundName);
    if (sit == set.namedSounds.end()) return;

    const std::string& path = sit->second;
    RegisterWavAsSample(soundName, path);

    int sampleId = g_registeredSamples[std::hash<std::string>{}(soundName)];
    if (sampleId >= 0) {
        cDMAudio::PlaySample(sampleId);
    }
}

static void UpdateTrainSounds(CVehicle* veh) {
    int vehKey = (int)(uintptr_t)veh;
    static std::map<int, TrainSoundState> &stateMap = g_trainSoundStates;
    TrainSoundState& state = stateMap[vehKey];

    if (!veh) return;

    bool engineOn = (veh->m_nEngineState == 1);
    bool braking = (veh->m_fBrakePedal > 0.1f);
    bool accelerating = (veh->m_fGasPedal > 0.1f);

    // Check doors open (use existing mod door logic)
    bool doorsOpen = false;
    // For simplicity, let's assume veh->m_nTrainDoorOpenLeft or Right flags (replace with your mod logic)
    if (veh->m_nTrainDoorOpenLeft || veh->m_nTrainDoorOpenRight) {
        doorsOpen = true;
    }

    if (engineOn != state.engineOn) {
        if (engineOn) PlayTrainSound(veh, "engine_start");
        else PlayTrainSound(veh, "engine_stop");
        state.engineOn = engineOn;
    }
    if (braking != state.braking) {
        if (braking) PlayTrainSound(veh, "brake_start");
        else PlayTrainSound(veh, "brake_release");
        state.braking = braking;
    }
    if (accelerating != state.accelerating) {
        if (accelerating) PlayTrainSound(veh, "accelerate");
        else PlayTrainSound(veh, "engine_idle");
        state.accelerating = accelerating;
    }
    if (doorsOpen != state.doorsOpen) {
        if (doorsOpen) PlayTrainSound(veh, "door_open");
        else PlayTrainSound(veh, "door_close");
        state.doorsOpen = doorsOpen;
    }
}

// --------------------
// Door and light animation thread
// --------------------

struct TrainInstance {
    CVehicle* veh;
    struct DoorState {
        float progress = 0.0f;
        bool opening = false;
        bool closing = false;
    };
    std::vector<DoorState> doors;
};

static std::map<CVehicle*, TrainInstance> g_trainInstances;

static void DoorLightThread() {
    while (g_threadRun) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        std::lock_guard<std::mutex> lock(g_mutex);

        for (auto &pair : g_trainInstances) {
            CVehicle* veh = pair.first;
            TrainInstance& ti = pair.second;

            // Update door progress smoothly
            for (auto &door : ti.doors) {
                if (door.opening) {
                    door.progress += 0.1f;
                    if (door.progress > 1.0f) door.progress = 1.0f;
                } else if (door.closing) {
                    door.progress -= 0.1f;
                    if (door.progress < 0.0f) door.progress = 0.0f;
                }
            }

            // Update train sounds per vehicle state
            UpdateTrainSounds(veh);

            // Set light status for the train car CJ is driving (both lights on)
            if (veh->m_nModelIndex == FindFreeModelIndex()) {
                // leave as-is or custom logic
            }
            // Add tail light logic here if needed...
        }
    }
}

// --------------------
// Plugin load and init
// --------------------

static void InitPlugin() {
    // Create INIs if missing
    EnsureIniFile(VEHICLES_INI, DEF_VEHICLES);
    EnsureIniFile(PEDS_INI, DEF_PEDS);
    EnsureIniFile(CARGRP_INI, DEF_CARGRP);
    EnsureIniFile(PEDGRP_INI, DEF_PEDGRP);
    EnsureIniFile(SOUNDS_INI, DEF_SOUNDS);
    EnsureIniFile(CONTROLS_INI, DEF_CONTROLS);

    // Load and register models
    LoadVehiclesINI();
    for (const auto &v : g_vehicles) RegisterVehicleModel(v);

    LoadPedsINI();
    for (const auto &p : g_peds) RegisterPedModel(p);

    RegisterCarGroups();
    RegisterPedGroups();

    LoadSoundsINI();

    // Start door/light thread
    g_threadRun = true;
    std::thread(DoorLightThread).detach();
}

// --------------------
// Plugin unload cleanup
// --------------------

static void ShutdownPlugin() {
    g_threadRun = false;
    ClearAllCustomModels();
}

// --------------------
// Exported plugin main entry
// --------------------

extern "C" __declspec(dllexport) void __cdecl PluginMain() {
    InitPlugin();
}

// --------------------
// Plugin cleanup on exit
// --------------------

extern "C" __declspec(dllexport) void __cdecl PluginShutdown() {
    ShutdownPlugin();
}

