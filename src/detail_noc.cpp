#include "detail_noc.hpp"

DetailNoC::DetailNoC(Simulator* sim, BookSimConfig const& booksim_cfg, const Config& cfg, Logger& logger,
                     uint64_t ticks_per_cycle, const std::vector<BaseModule*>& core_list,
                     const std::vector<BaseModule*>& mc_list)
    : BaseModule(sim, cfg, logger, ticks_per_cycle, -1),
      m_num_cores(cfg.core_num),
      core_ptrs(core_list),
      mc_ptrs(mc_list) {
    switch (cfg.core_num) {
        case 1:
            m_noc_mesh_width = 2;
            m_noc_mesh_height = 2;
            m_base_ddr_id = 2;
            m_core_num_per_row = 1;
            break;
        case 2:
            m_noc_mesh_width = 2;
            m_noc_mesh_height = 2;
            m_base_ddr_id = 2;
            m_core_num_per_row = 2;
            break;
        case 4:
            m_noc_mesh_width = 3;
            m_noc_mesh_height = 3;
            m_base_ddr_id = 6;
            m_core_num_per_row = 2;
            break;
        case 8:
            m_noc_mesh_width = 5;
            m_noc_mesh_height = 5;
            m_base_ddr_id = 20;
            m_core_num_per_row = 2;
            break;
        case 16:
            m_noc_mesh_width = 5;
            m_noc_mesh_height = 5;
            m_base_ddr_id = 20;
            m_core_num_per_row = 4;
            break;
        case 32:
            m_noc_mesh_width = 10;
            m_noc_mesh_height = 10;
            m_base_ddr_id = 40;
            m_core_num_per_row = 8;
            break;
        case 64:
            m_noc_mesh_width = 10;
            m_noc_mesh_height = 10;
            m_base_ddr_id = 80;
            m_core_num_per_row = 8;
            break;
        default:
            std::cerr << "Error: Invalid core number" << std::endl;
            exit(1);
    }
    link_width_bytes = cfg.noc_link_width;
    m_request_queue_depth = cfg.noc_request_queue_depth;
    ni_request_queues.resize(m_noc_mesh_width * m_noc_mesh_height);

    int total_tiles = m_noc_mesh_width * m_noc_mesh_height;
    tile_to_module.resize(total_tiles, nullptr);

    for (auto& core : core_ptrs) {
        int virtual_core_id = core->getId();
        int physical_core_id =
            (virtual_core_id / m_core_num_per_row) * m_noc_mesh_width + (virtual_core_id % m_core_num_per_row);
        assert(physical_core_id < total_tiles);
        tile_to_module[physical_core_id] = core;
    }

    for (auto& mc : mc_ptrs) {
        int tile_id = mc->getId() + m_base_ddr_id;
        assert(tile_id < total_tiles);
        tile_to_module[tile_id] = mc;
    }

    // Booksim Config
    vector<Network*> net;  // Create the network
    int subnets = booksim_cfg.GetInt("subnets");
    net.resize(subnets);

    for (int i = 0; i < subnets; ++i) {
        std::string net_name = "network_" + to_string(i);
        net[i] = Network::New(booksim_cfg, net_name);
    }
    traffic_manager = TrafficManager::New(booksim_cfg, net);

    logger.log("NoC Frequency: %f GHz", cfg.noc_freq);
    logger.log("Link Width: %d bytes", link_width_bytes);
    logger.log("Request Queue Depth: %d", m_request_queue_depth);
    logger.log("Detailed NoC initialized!");
}

void DetailNoC::scheduleNextEvent() {
    uint64_t next_tick = m_cycle * m_ticks_per_cycle;
    Event next_e;
    next_e.tick = next_tick;
    next_e.callback = [this]() { tick(); };
    sim->schedule(next_e);
}

Flit* DetailNoC::getEnjectedFlit() {
    if (_pending_packets.empty()) return nullptr;
    Flit* f = _pending_packets.front();
    _pending_packets.pop();
    return f;
}

void DetailNoC::tick() {
    ScopedTimer timer(&sim->profile.noc_time_ns);

    while (!_pending_packets.empty()) {
        Flit* f = _pending_packets.front();
        _pending_packets.pop();
        f->Free();
        cout << "Warning: Unprocessed packet " << f->pid << " from previous cycle discarded" << endl;
    }

    traffic_manager->_Step();
    std::queue<Flit*> new_packets = traffic_manager->takeCompletedPackets();

    _pending_packets = std::move(new_packets);

    logger.log("[Tick %d-NoC Cycle %d] NoC is ticked", sim->getTick(), getCycle());

    Flit* f;

    while ((f = getEnjectedFlit()) != nullptr) {
        logger.log("[Tick %d-NoC Cycle %d] Flit %d ejected from NoC (src %d, dst %d)", sim->getTick(), getCycle(),
                   f->id, f->src, f->dest);
        FlitType flit_type = f->type;
        int src_tile_id = f->src;
        int dst_tile_id = f->dest;

        BaseModule* dst_mod = tile_to_module[dst_tile_id];
        /*
        bool find_success_flg = false;
        // get dst module pointer according to dst tile id
        // find in ddr channels
        for (auto& mc : mc_ptrs) {
            if ((mc->getId() + m_base_ddr_id) == dst_tile_id) {
                dst_mod = mc;
                find_success_flg = true;
                break;
        }
        // find in core tile array
        if (!find_success_flg) {
            for (auto& core : core_ptrs) {
                int virtual_core_id = core->getId();
                int physical_core_id =
                    (virtual_core_id / m_core_num_per_row) * m_noc_mesh_width + (virtual_core_id % m_core_num_per_row);
                assert(physical_core_id < m_noc_mesh_width * m_noc_mesh_height);
                if (physical_core_id == dst_tile_id) {
                    dst_mod = core;
                    break;
                }
            }
        }
        */

        if (dst_mod == nullptr) {
            logger.log("[Tick %d-NoC Cycle %d] NoC No module found for tile %d", sim->getTick(), getCycle(),
                       dst_tile_id);
            throw std::runtime_error("No module found for tile " + std::to_string(dst_tile_id));
        }

        Message msg;
        if (flit_type == FlitType::READ_REQUEST) {
            msg.type = MsgType::LD_REQ;
            msg.data_size = 1;
            msg.request_data_size = f->data_size;
        } else if (flit_type == FlitType::WRITE_REQUEST) {
            msg.type = MsgType::ST_REQ;
            msg.data_size = f->data_size;
            msg.request_data_size = 1;
        } else if (flit_type == FlitType::READ_REPLY) {
            msg.type = MsgType::LD_RESP;
            msg.data_size = f->data_size;
            msg.request_data_size = 1;
        } else if (flit_type == FlitType::WRITE_REPLY) {
            msg.type = MsgType::ST_RESP;
            msg.data_size = 1;
            msg.request_data_size = f->data_size;
        }
        msg.dst_addr = f->dest_addr;
        msg.src_addr = f->src_addr;
        // generate a receive event and schedule it in the next noc cycle
        Event recv_e;
        recv_e.tick = sim->getTick() + m_ticks_per_cycle;
        recv_e.callback = [dst_mod, msg]() mutable { dst_mod->recvMsg(msg); };

        sim->schedule(recv_e);
        logger.log(
            "Tick %d NoC Cycle %d: NoC Scheduled recvMsg event [from tile%d to: tile%d], will call recvMsg at tick %d",
            sim->getTick(), getCycle(), src_tile_id, dst_tile_id, recv_e.tick);
        f->Free();
    }

    /*for (int i = 0; i < m_noc_mesh_height * m_noc_mesh_width; i++) {
        if (!ni_request_queues[i].empty()) {
            Message msg = ni_request_queues[i].front();
            sendMsg(msg);
            ni_request_queues[i].pop();
        }
    }*/

    int n = active_routers.size();

    for (int i = 0; i < n; i++) {
        int rid = active_routers.front();
        active_routers.pop();

        Message msg = ni_request_queues[rid].front();
        ni_request_queues[rid].pop();

        sendMsg(msg);

        if (!ni_request_queues[rid].empty()) active_routers.push(rid);
    }

    if (m_cycle * m_ticks_per_cycle <= sim->getTick()) {
        m_cycle++;
        scheduleNextEvent();  // CA NoC model需要自调度下一个周期的tick事件
    }
}

void DetailNoC::sendMsg(Message& msg) {
    // get src/dst tile id from msg
    int src_tile_id = parseAddr(msg.src_addr, m_num_cores);
    int dst_tile_id = parseAddr(msg.dst_addr, m_num_cores);
    logger.log("[Tick %d-NoC Cycle %d] NoC Sending message (flit id=%d) from tile %d to tile %d", sim->getTick(),
               getCycle(), cur_pkt_id * MAX_FLITS_PER_PACKET, src_tile_id, dst_tile_id);

    // generate Booksim packet and inject to Booksim
    FlitType f_type;
    int pkt_size_in_flits = 1;  // number of flits in the packet
    int req_data_size = 1;      // requested data size
    if (msg.type == MsgType::LD_REQ) {
        f_type = FlitType::READ_REQUEST;
        req_data_size = msg.request_data_size;
    } else if (msg.type == MsgType::ST_REQ) {
        f_type = FlitType::WRITE_REQUEST;
        // convert data_size to number of flits
        pkt_size_in_flits = (msg.data_size + link_width_bytes - 1) / link_width_bytes;
    } else if (msg.type == MsgType::LD_RESP) {
        f_type = FlitType::READ_REPLY;
        pkt_size_in_flits = (msg.request_data_size + link_width_bytes - 1) / link_width_bytes;
    } else if (msg.type == MsgType::ST_RESP) {
        f_type = FlitType::WRITE_REPLY;
        pkt_size_in_flits = 1;
    }

    for (int i = 0; i < pkt_size_in_flits; i++) {
        Flit* f = Flit::New();
        f->id = cur_pkt_id * MAX_FLITS_PER_PACKET + i;
        f->pid = cur_pkt_id;
        f->src = src_tile_id;
        f->dest = dst_tile_id;
        f->data_size = req_data_size;
        f->src_addr = msg.src_addr;
        f->dest_addr = msg.dst_addr;
        f->ctime = m_cycle;
        f->type = f_type;
        f->cl = 0;
        f->pri = 0;
        f->vc = -1;
        f->subnetwork = f_type;
        f->watch = false;
        f->record = false;

        if (i == 0) {  // Head Flit
            f->head = true;
        } else {
            f->head = false;
        }

        if (i == (pkt_size_in_flits - 1)) {  // Tail Flit
            f->tail = true;
        } else {
            f->tail = false;
        }

        traffic_manager->_total_in_flight_flits[f->cl].insert(std::make_pair(f->id, f));
        traffic_manager->_partial_packets[f->src][f->cl].push_back(f);
    }
    cur_pkt_id++;
}

bool DetailNoC::recvMsg(Message& msg) {
    int inj_router_id = parseAddr(msg.src_addr, m_num_cores);
    if (ni_request_queues[inj_router_id].size() >= m_request_queue_depth) {
        logger.log("[Tick %d-NoC Cycle %d] NoC Router %d: request queue is full, backpressure asserted", sim->getTick(),
                   getCycle(), inj_router_id);
        return false;
    } else {
        if (ni_request_queues[inj_router_id].empty()) {
            active_routers.push(inj_router_id);
        }
        ni_request_queues[inj_router_id].push(msg);
        logger.log("[Tick %d-NoC Cycle %d] NoC Router %d received msg", sim->getTick(), getCycle(), inj_router_id);
        return true;
    }
}

TrafficManager* DetailNoC::getTrafficManager() {
    return traffic_manager;
}