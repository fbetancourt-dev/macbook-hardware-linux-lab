# Hardware Subsystem: Intel DSL5520 Thunderbolt 2 (PCIe Hotplug)

The `MacBookPro11,3` incorporates an **Intel DSL5520 Falcon Ridge 4C** Thunderbolt 2 controller providing dual 20 Gbps bi-directional channels over two Mini DisplayPort connectors.

---

## 💻 Hardware Topology

* **Controller:** Intel DSL5520 Falcon Ridge 4C (`8086:156d` / `8086:156c`)
* **Upstream Link:** PCIe Gen2 x4 (20 Gbps) off PCH Root Port #3 (`00:1c.2`)
* **Downstream Ports:** Dual Thunderbolt 2 ports supporting DisplayPort 1.2 pass-through and PCIe tunneling.
* **Kernel Driver:** `drivers/thunderbolt/` (`thunderbolt.ko`) with native Linux PCIe hotplug support (`pciehp`).
* **Security Levels:** Configurable in Linux via `/sys/bus/thunderbolt/devices/domain0/security` (`none`, `user`, `secure`, `dponly`).
